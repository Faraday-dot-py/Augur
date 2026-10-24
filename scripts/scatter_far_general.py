"""Far-kernel generalisation across interaction laws (docs/debugging/scatter-far-kernel-generalization.md).

A law is a signed inward pair force F(r) = exp(-r/lam) * (a1 r/(r^2+eps^2)^((p1+1)/2) + a2 r/(r^2+eps^2)^((p2+1)/2)); positive = attraction.
Gravity = (a1 1, p1 2, eps .5). LJ (sigma 2, eps_LJ 1) = a1 -24/sigma*2 sigma^13 r^-13 + 24/sigma sigma^7 r^-7 (unsoftened). Labels are evaluated in fp64.
Modes:
  base  <name,...>   per-interaction retrained plain MLP (decision (a); R1b recipe/size, init from the shipped kmlp)
  cond  --mode theta|ctx --seed S   kernel conditioned on theta (b) or on force samples at r in [16, 32] (c); trained on non-held-out laws only
  eval               kernel-level metrics for every test law and method, scene-level mixed-sign direct-sum check, JSON out
  bake               write gravity-bound kernel files for the end-to-end harness (scatter_far.load_kernel)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from scripts import scatter_far as sfar
from scripts.scatter_tiling_eval import load

EPS = 0.5
SIG = 2.0
FLOOR = 0.02
PROBE = np.geomspace(16.0, 32.0, 8)

# name: (a1, p1, a2, p2, inv_lam, eps, decaying)
LAWS = {
    "pow1": (1.0, 1.0, 0.0, 2.0, 0.0, EPS, False),
    "pow1.5": (1.0, 1.5, 0.0, 2.0, 0.0, EPS, False),
    "pow2": (1.0, 2.0, 0.0, 2.0, 0.0, EPS, False),
    "pow3": (1.0, 3.0, 0.0, 2.0, 0.0, EPS, False),
    "yuk8": (1.0, 2.0, 0.0, 2.0, 1 / 8, EPS, True),
    "yuk32": (1.0, 2.0, 0.0, 2.0, 1 / 32, EPS, True),
    "yuk128": (1.0, 2.0, 0.0, 2.0, 1 / 128, EPS, True),
    "lj": (-24 / SIG * 2 * SIG ** 13, 13.0, 24 / SIG * SIG ** 7, 7.0, 0.0, 0.0, True),
    "rep2": (-1.0, 2.0, 0.0, 2.0, 0.0, EPS, False),
    "nr1b": (0.5, 3.0, 0.0, 2.0, 0.0, EPS, False),
    "scr32": (-1.0, 2.0, 0.0, 2.0, 1 / 32, EPS, True),
    "nr2b": (0.5, 1.5, 0.0, 2.0, 0.0, EPS, False),
}
HELD = {"pow1.5", "yuk32", "scr32", "nr2b"}
TRAIN_SHAPES = ["pow1", "pow2", "pow3", "yuk8", "yuk128", "lj"]
DESC = {"pow1": "power p=1", "pow1.5": "power p=1.5 (held out)", "pow2": "power p=2 (gravity)", "pow3": "power p=3", "yuk8": "Yukawa lam=8",
        "yuk32": "Yukawa lam=32 (held out)", "yuk128": "Yukawa lam=128", "lj": "LJ sigma 2", "rep2": "like charges / pred->prey (repulsive p=2)",
        "nr1b": "prey->pred (attractive p=3, a .5)", "scr32": "screened repulsive lam=32 (held out)", "nr2b": "prey->pred (attractive p=1.5, a .5; held out)"}
# test pairs: charges and non-reciprocal laws are ordered-pair channels
GROUPS = {"powerlaw": ["pow1", "pow1.5", "pow2", "pow3"], "yukawa": ["yuk8", "yuk32", "yuk128"], "charges": ["pow2", "rep2", "scr32"],
          "lj": ["lj"], "nonrecip1": ["rep2", "nr1b"], "nonrecip2": ["scr32", "nr2b"]}


def theta_of(law, scale=1.0):
    a1, p1, a2, p2, il, eps, _ = law
    sl = lambda a: math.copysign(math.log10(1 + abs(a)), a)
    return [sl(a1 * scale), p1 / 4, sl(a2 * scale), p2 / 4, 32 * il]


def force_t(r, law, scale=1.0):
    a1, p1, a2, p2, il, eps, _ = law
    r = r.double()
    if eps == 0.0:
        r = r.clamp_min(1.0)
    d = r * r + eps * eps
    F = a1 * scale * r / d ** ((p1 + 1) / 2) + (a2 * scale * r / d ** ((p2 + 1) / 2) if a2 != 0 else 0.0)
    return F * torch.exp(-r * il)


def amp(law):
    return 1.0 if law[6] and law[5] == 0.0 else max(abs(law[0]), abs(law[2]))


def floor_of(r, law, scale=1.0):
    return (FLOOR * abs(scale) * amp(law) * sfar.f_true(r.double())) if law[6] else torch.zeros_like(r.double())


def ctx_of(law, scale=1.0, noise=0.0, gen=None):
    r = torch.tensor(PROBE, dtype=torch.float64)
    x = force_t(r, law, scale) / sfar.f_true(r)
    if noise > 0:
        x = x * (1 + noise * torch.randn(x.shape, generator=gen, dtype=torch.float64))
    return (torch.sign(x) * torch.log10(1 + 1e3 * x.abs()) / 2).tolist()


def force_net(mod, r, c=None):
    r = r.clone().requires_grad_(True)
    K = mod(r) if c is None else mod.forward_c(r, c)
    F, = torch.autograd.grad(K.sum(), r, create_graph=True)
    return K, F


def rel_loss(F, Ft, fl, r):
    far = (r >= 2.0).to(F.dtype)
    h = (r >= 4.0).to(F.dtype)
    den = Ft.abs() + fl + 1e-30
    e = torch.asinh((F - Ft) / den)
    hinge = torch.relu(-torch.sign(Ft) * F / den).clamp(max=5)
    return ((e ** 2 * far).sum(-1) / far.sum(-1) + (hinge * h).sum(-1) / h.sum(-1)).mean()


def sample_r(n, dev, rmax=1500.0, near=1 / 4):
    nn_ = int(n * near)
    return torch.cat([torch.rand(nn_, device=dev) * 34.0, 10 ** (torch.rand(n - nn_, device=dev) * math.log10(rmax))]).clamp_min(1e-3)


def zone_metrics(F, name, scale, r, ref=None):
    """F: net force (float64, same r). Returns pass-criteria metrics for one law."""
    law = LAWS[name]
    Ft = force_t(r, law, scale)
    fl = floor_of(r, law, scale)
    in_rng = (r >= 4) & (r <= 512)
    Z = in_rng & (Ft.abs() >= fl) if law[6] else in_rng
    ratio = F / Ft
    out = {"dev": float((ratio[Z] - 1).abs().max()), "rmin": float(ratio[Z].min()), "rmax": float(ratio[Z].max()),
           "sign_ok": float((torch.sign(F[Z]) == torch.sign(Ft[Z])).double().mean()), "n_zone": int(Z.sum())}
    if law[6]:
        T = (r >= 4) & (r <= 1500) & ~Z & (r > 4)
        S = abs(scale) * amp(law) * sfar.f_true(r)
        out["tail"] = float((F[T].abs() / S[T]).max()) if T.any() else 0.0
    if ref is not None:
        Zi = (r >= 4) & (r <= 32) & (ref.abs() >= fl)
        out["interior"] = float(((F[Zi] / ref[Zi]) - 1).abs().max()) if Zi.any() else 0.0
    out["pass"] = bool(out["dev"] <= 0.1 and out["sign_ok"] == 1.0 and out.get("tail", 0.0) <= 0.05 and out.get("interior", 0.0) <= 0.05)
    out["kill"] = bool(out["rmin"] < 0.5 or out["rmax"] > 2.0)
    return out


def train_base(a):
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    m = load(a.ckpt, dev)
    for name in a.laws.split(","):
        if os.path.exists(f"{a.outdir}/base_{name}.pt"):
            continue
        law = LAWS[name]
        mod = sfar.MLPKernel(pp=m.pp, split=m.split).to(dev)
        mod.net.load_state_dict(m.kmlp.state_dict())
        opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
        val_r = torch.logspace(0, math.log10(1500.0), 6000, device=dev)
        best = (1e9, None)
        t0 = time.time()
        for it in range(a.steps + 1):
            for g in opt.param_groups:
                g["lr"] = 1e-3 * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / a.steps)))
            r = sample_r(6144, dev)
            Ft = force_t(r, law).float()
            fl = floor_of(r, law).float()
            _, F = force_net(mod, r)
            loss = rel_loss(F, Ft, fl, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 1000 == 0:
                _, Fv = force_net(mod, val_r)
                mt = zone_metrics(Fv.detach().double(), name, 1.0, val_r.double())
                score = mt["dev"] + mt.get("tail", 0.0) + (0 if mt["sign_ok"] == 1.0 else 10)
                print(f"{name} it {it} loss {float(loss.detach()):.3e} dev {mt['dev']:.3f} tail {mt.get('tail', 0):.3f} sign {mt['sign_ok']:.3f} t {time.time() - t0:.0f}s", flush=True)
                if score < best[0]:
                    best = (score, {k: v.detach().clone() for k, v in mod.state_dict().items()}, it)
        torch.save({"arch": "mlp", "kw": {"pp": m.pp, "split": m.split}, "state": best[1], "meta": {"law": name, "best_it": best[2], "score": best[0]}},
                   f"{a.outdir}/base_{name}.pt")
        print(name, "best it", best[2], "score", best[0], flush=True)


def train_cond(a):
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    m = load(a.ckpt, dev)
    old = sfar.old_fn(m)
    dc = 5 if a.mode == "theta" else len(PROBE)
    mod = sfar.CondKernel(dc=dc, hidden=a.hidden, pp=m.pp, split=m.split).to(dev)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
    B, G, n = a.batch, a.grav, a.npts
    t0 = time.time()
    ctx_cache = {}
    for it in range(a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / a.steps)))
        names = [TRAIN_SHAPES[i] for i in np.random.randint(0, len(TRAIN_SHAPES), B)]
        scales = np.sign(np.random.rand(B) - 0.5) * 10 ** np.random.uniform(-1, 1, B)
        laws = [(LAWS[nm], float(s)) for nm, s in zip(names, scales)]
        laws += [(LAWS["pow2"], 1.0)] * G
        cs = torch.tensor([theta_of(l, s) if a.mode == "theta" else ctx_of(l, s) for l, s in laws], dtype=torch.float32, device=dev)
        r = torch.stack([sample_r(n, dev) for _ in range(B + G)])
        Ft = torch.stack([force_t(r[i], l, s) for i, (l, s) in enumerate(laws)]).float()
        fl = torch.stack([floor_of(r[i], l, s) for i, (l, s) in enumerate(laws)]).float()
        K, F = force_net(mod, r, cs)
        loss = rel_loss(F[:B], Ft[:B], fl[:B], r[:B])
        rg = r[B:]
        Fo = sfar.force_of(old, rg.reshape(-1)).reshape(rg.shape)
        w = sfar.smoothstep((rg - 28.0) / 20.0) if hasattr(sfar, "smoothstep") else ((rg - 28.0) / 20.0).clamp(0, 1)
        Fg = (1 - w) * Fo + w * sfar.f_true(rg)
        rg_far = rg >= 4
        rel = torch.asinh((F[B:] - Fg) / Fg.clamp_min(1e-12))
        l_g = (rel[rg_far] ** 2).mean() + ((F[B:] - Fg)[~rg_far] ** 2).mean()
        near = rg <= 28.0
        l_k = ((K[B:] - old(rg).detach())[near] ** 2).mean()
        loss = loss + l_g + 100.0 * l_k
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if it % 2000 == 0:
            print(f"it {it} loss {float(loss.detach()):.3e} grav-rel {float(l_g.detach()):.3e} k {float(l_k.detach()):.3e} t {time.time() - t0:.0f}s", flush=True)
        if it % 10000 == 0 and it:
            torch.save({"arch": "cond", "kw": {"dc": dc, "hidden": a.hidden, "pp": m.pp, "split": m.split}, "state": mod.state_dict(), "meta": {"args": vars(a), "it": it}},
                       f"{a.outdir}/{a.mode}_s{a.seed}.pt")
    torch.save({"arch": "cond", "kw": {"dc": dc, "hidden": a.hidden, "pp": m.pp, "split": m.split}, "state": mod.state_dict(), "meta": {"args": vars(a), "it": a.steps}},
               f"{a.outdir}/{a.mode}_s{a.seed}.pt")


def smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


sfar.smoothstep = smoothstep


def kernel_force(mod, r, c=None):
    if c is not None:
        mod.c = torch.tensor(c, dtype=torch.float32, device=r.device)
    r = r.clone().float().requires_grad_(True)
    with torch.enable_grad():
        K = mod(r)
        F, = torch.autograd.grad(K.sum(), r)
    return F.detach().double()


def load_k(path, dev):
    return sfar.load_kernel(path, dev)


def run_eval(a):
    dev = torch.device("cuda")
    r = torch.logspace(0, math.log10(1500.0), 8000, device=dev, dtype=torch.float64)
    res = {"laws": {}}
    base_ref = {}
    seeds = [int(s) for s in a.seeds.split(",")]
    ck = {}
    for nm in LAWS:
        path = f"{a.dir}/base_{nm}.pt" if nm != "pow2" else a.r1b
        ck[nm] = load_k(path, dev)
        base_ref[nm] = kernel_force(ck[nm], r)
    cond = {(mode, s): load_k(f"{a.dir}/{mode}_s{s}.pt", dev) for mode in ("theta", "ctx") for s in seeds}
    gen = torch.Generator().manual_seed(4738)
    for nm, law in LAWS.items():
        row = {"desc": DESC[nm], "held_out": nm in HELD, "baseline": zone_metrics(base_ref[nm], nm, 1.0, r)}
        row["baseline"] = zone_metrics(base_ref[nm], nm, 1.0, r, ref=None)
        for mode in ("theta", "ctx"):
            rows = []
            for s in seeds:
                F = kernel_force(cond[(mode, s)], r, theta_of(law) if mode == "theta" else ctx_of(law))
                mt = zone_metrics(F, nm, 1.0, r, ref=base_ref[nm])
                if mode == "ctx":
                    nz = {}
                    for noise in (0.02, 0.10):
                        devs = []
                        for _ in range(5):
                            Fn = kernel_force(cond[(mode, s)], r, ctx_of(law, noise=noise, gen=gen))
                            devs.append(zone_metrics(Fn, nm, 1.0, r)["dev"])
                        nz[str(noise)] = float(np.max(devs))
                    mt["noisy_ctx_worst_dev"] = nz
                ratio_at = {str(x): float(F[int(torch.argmin((r - x).abs()))] / force_t(r[int(torch.argmin((r - x).abs()))][None], law)[0]) for x in (4, 8, 16, 32, 64, 128, 256, 512)}
                mt["ratio_at"] = ratio_at
                rows.append(mt)
            row[mode] = rows
        res["laws"][nm] = row
        print(nm, json.dumps({k: (v if not isinstance(v, list) else [(x["dev"], x.get("tail"), x["sign_ok"], x["interior"]) for x in v]) for k, v in row.items() if k != "desc"}), flush=True)
    res["scene"] = scene_check(ck, cond, seeds, dev)
    json.dump(res, open(a.out, "w"), indent=1)


@torch.no_grad()
def scene_check(ck, cond, seeds, dev, n=1500, L=512.0, seed=4738):
    """Mixed-sign scene: n particles with random +-1 charges in a [-L/2, L/2]^2 box, pair force s_ij * f(r) with s_ij = -q_i q_j (like repels)
    for the named law, summed over pairs r >= 4 in fp64 (exact) vs the same sum with the kernel's pair force for a = s_ij."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = ((torch.rand(n, 2, generator=g, dtype=torch.float64) - 0.5) * L).to(dev)
    q = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).double().to(dev)
    rel = pos[None, :, :] - pos[:, None, :]
    d = rel.norm(dim=-1)
    sij = -q[:, None] * q[None, :]
    mask = (d >= 4) & (d <= 1500)
    out = {}
    for nm in ("pow2", "pow1.5", "yuk32", "pow3", "yuk8"):
        law = LAWS[nm]
        dd = d[mask]
        Ft = force_t(dd, law)
        def total(fpos, fneg):
            f = torch.zeros_like(d)
            f[mask] = torch.where(sij[mask] > 0, fpos, fneg)
            return (f[..., None] * rel / d.clamp_min(1e-9)[..., None]).sum(1)
        ex = total(Ft, -Ft)
        den = ex.norm(dim=-1).sum()
        row = {}
        for label, getf in (("baseline", lambda: (kernel_force(ck[nm], dd), kernel_force(ck[nm], dd))),):
            fp = kernel_force(ck[nm], dd)
            row[label] = float((total(fp, -fp) - ex).norm(dim=-1).sum() / den)
        for mode in ("theta", "ctx"):
            errs = []
            for s in seeds:
                mod = cond[(mode, s)]
                fp = kernel_force(mod, dd, theta_of(law) if mode == "theta" else ctx_of(law))
                fn = kernel_force(mod, dd, theta_of(law, -1.0) if mode == "theta" else ctx_of(law, -1.0))
                errs.append(float((total(fp, fn) - ex).norm(dim=-1).sum() / den))
            row[mode] = errs
        out[nm] = row
        print("scene", nm, json.dumps(row), flush=True)
    return out


def bake(a):
    dev = torch.device("cuda")
    seeds = [int(s) for s in a.seeds.split(",")]
    law = LAWS["pow2"]
    gen = torch.Generator().manual_seed(4738)
    for mode in ("theta", "ctx"):
        for s in seeds:
            st = torch.load(f"{a.dir}/{mode}_s{s}.pt", map_location="cpu", weights_only=False)
            c = theta_of(law) if mode == "theta" else ctx_of(law)
            st["state"]["c"] = torch.tensor(c, dtype=torch.float32)
            torch.save(st, f"{a.dir}/grav_{mode}_s{s}.pt")
            if mode == "ctx":
                st["state"]["c"] = torch.tensor(ctx_of(law, noise=0.02, gen=gen), dtype=torch.float32)
                torch.save(st, f"{a.dir}/grav_ctxnoise2_s{s}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["base", "cond", "eval", "bake"])
    ap.add_argument("--ckpt", default="kern/E_frozen.pt")
    ap.add_argument("--outdir", default="kern/gen")
    ap.add_argument("--dir", default="kern/gen")
    ap.add_argument("--laws", default="")
    ap.add_argument("--mode", default="theta")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--seeds", default="4738,4739,4740")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grav", type=int, default=8)
    ap.add_argument("--npts", type=int, default=256)
    ap.add_argument("--r1b", default="results/far_kernels/r1b_mlp.pt")
    ap.add_argument("--out", default="results/far_general_eval.json")
    a = ap.parse_args()
    {"base": train_base, "cond": train_cond, "eval": run_eval, "bake": bake}[a.what](a)


if __name__ == "__main__":
    main()
