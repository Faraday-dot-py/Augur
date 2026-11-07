"""Training-data generators for the heterogeneous tiered architecture (docs/theory/heterogeneous-tiered.md, "Training data and
evaluation scenes"). All truth is computed on GPU in fp64; shards are npz files written atomically, named by (kind, seed, shard),
and existing shards are skipped on --resume (SIGTERM finishes the current shard and exits).

  python scripts/tiered_gen.py tier0   --kinds all --shards 4 --scenes 64 --steps 32 --seed 4738 --out results/tiered/data
  python scripts/tiered_gen.py harvest --shards 4 --windows 32 --steps 16 --seed 4738 --out results/tiered/data
  python scripts/tiered_gen.py tier1   --shards 4 --scenes 8 --span 128 --lower range8 --states truth --out results/tiered/data
  python scripts/tiered_gen.py truthcheck

tier0: per-scene smooth external field (ExtField: value/Jacobian/Hessian of the acceleration, optional cubic term and slow time
variation); the state evolves under near-field law + external field; target acc_near = truth - external (known exactly).
harvest: windows (tile+halo = 64) cut from a large softened-gravity rollout; the external field is the direct-sum far-field
Taylor expansion about the window centre from every particle outside the window at that frame; acc_far_exact is also stored so
the expansion error is measurable; per-frame masks follow particles in and out of the window.
tier1: large scenes, frozen lower-tier hook, target = truth - lower output (direct sum up to --direct-max, cell multipole above)."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from scripts import tiered_iface as ti
from scripts import tiered_scenes as sc
from scripts import tiered_truth as tt

STOP = False
REF_ACC = dict(grav=0.3, lj=1.0, contact=3.0, bonded=1.0)


def _sigterm(*_):
    global STOP
    STOP = True
    print("SIGTERM: finishing current shard", flush=True)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def seed_for(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def save_atomic(path, **arrs):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def batch_key(s):
    return f"{s['kind']}.{'+'.join(sorted(s['law']))}.{s['bound']['kind']}.{s['dt']}.{s['sub']}"


def ref_accel(s):
    for k in ("lj", "contact", "bonded", "grav"):
        if k in s["law"]:
            return REF_ACC[k]
    return 1.0


def sample_ext(B, gen, dev, ref, p_ext, high_order, p_rate):
    on = (torch.rand(B, generator=gen, device=dev, dtype=torch.float64) < p_ext).to(torch.float64)
    ho = (torch.rand(B, generator=gen, device=dev, dtype=torch.float64) < high_order).to(torch.float64)
    rt = (torch.rand(B, generator=gen, device=dev, dtype=torch.float64) < p_rate).to(torch.float64)
    e = tt.ExtField.random(B, gen, dev, a_scale=ref, j_scale=ref / 30.0, q_scale=ref / 600.0, rate_scale=0.02)
    return tt.ExtField(e.a0 * on[:, None], e.J * on[:, None, None], e.Q * (on * ho)[:, None, None, None], e.rate * rt)


def tier0_batch(scenes, steps, gen, dev, args):
    b = sc.build_batch(scenes, dev)
    if b["tags"][0].get("equil"):
        sc.equilibrate(b, args.equil)
    ext = sample_ext(len(scenes), gen, dev, ref_accel(scenes[0]), args.p_ext, args.high_order, args.p_rate)
    r = tt.rollout(b["phys"], b["pos"], b["vel"], b["mass"], b["mask"], steps, b["dt"], b["sub"], ext=ext)
    B, K1, N, _ = r["pos"].shape
    E = torch.stack([b["phys"].energy(r["pos"][:, k], r["vel"][:, k], b["mass"], b["mask"]) for k in range(K1)], 1)
    ok = torch.isfinite(r["pos"]).all((1, 2, 3)) & torch.isfinite(r["acc_near"]).all((1, 2, 3)) & (r["vel"].abs().amax((1, 2, 3)) < 1e3)
    out = dict(pos=r["pos"], vel=r["vel"], acc_near=r["acc_near"], acc_ext=r["acc_ext"])
    if args.noise_pos > 0:
        pn = r["pos"] + args.noise_pos * torch.randn(r["pos"].shape, generator=gen, device=dev, dtype=torch.float64) * b["mask"][:, None, :, None]
        an = torch.stack([b["phys"].acc(pn[:, k], r["vel"][:, k], b["mass"], b["mask"]) for k in range(K1)], 1)
        out.update(pos_noisy=pn, acc_near_noisy=an)
    arrs = {k: v.float().cpu().numpy() for k, v in out.items()}
    arrs.update(mass=b["mass"].float().cpu().numpy(), mask=b["mask"].float().cpu().numpy(), kmat=b["phys"].kmat.float().cpu().numpy(),
                occ=b["occ"].cpu().numpy(), energy=E.cpu().numpy(), valid=ok.cpu().numpy(),
                ext_a0=ext.a0.cpu().numpy(), ext_J=ext.J.cpu().numpy(), ext_Q=ext.Q.cpu().numpy(), ext_rate=ext.rate.cpu().numpy(),
                dt=np.float64(b["dt"]), sub=np.int64(b["sub"]))
    tags = [dict(t) for t in b["tags"]]
    for i, t in enumerate(tags):
        t.pop("equil", None)
        if b["kind"] == "edge":
            c0 = (r["pos"][i, 0].abs() > sc.TILE / 2).any(-1)
            c1 = (r["pos"][i, -1].abs() > sc.TILE / 2).any(-1)
            t["cross_frac"] = float(((c0 != c1).float() * b["mask"][i]).sum() / b["mask"][i].sum())
        t["bound"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in b["bound"][i].items()}
    return arrs, tags


def gen_tier0(args, dev):
    kinds = sc.TIER0_KINDS if args.kinds == "all" else args.kinds.split(",")
    os.makedirs(args.out, exist_ok=True)
    for kind in kinds:
        for sh in range(args.shards):
            if STOP:
                return
            tag = f"t0_{kind}_s{args.seed}_{sh:03d}"
            if args.resume and any(f.startswith(tag) for f in os.listdir(args.out)):
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed_for(args.seed, kind, sh))
            gen = torch.Generator(device=dev).manual_seed(seed_for(args.seed, kind, sh, "ext"))
            scenes = [sc.make_scene(kind, rng, args.n_max) for _ in range(args.scenes)]
            groups = {}
            for s in scenes:
                groups.setdefault(batch_key(s), []).append(s)
            for key, g in groups.items():
                for c0 in range(0, len(g), args.batch):
                    arrs, tags = tier0_batch(g[c0:c0 + args.batch], args.steps, gen, dev, args)
                    meta = dict(kind=kind, key=key, seed=args.seed, shard=sh, steps=args.steps, git=git_rev(), tags=tags,
                                noise_pos=args.noise_pos, window=sc.WINDOW, tile=sc.TILE, halo=sc.HALO)
                    save_atomic(os.path.join(args.out, f"{tag}_{key.replace('.', '-')}_{c0:04d}.npz"), meta=np.array(json.dumps(meta)), **arrs)
                    print(f"{tag} {key} B={len(g[c0:c0 + args.batch])} N={arrs['pos'].shape[2]} valid={arrs['valid'].mean():.2f} {time.time() - t0:.1f}s", flush=True)


def pick_ic(rng):
    return str(rng.choice(["clumpy", "plummer", "disc", "cold_collapse", "uniform"]))


def harvest_one(args, dev, seed, sh):
    rng = np.random.default_rng(seed_for(seed, "harvest", sh))
    dom, nb, K = args.domain, args.n_big, args.steps
    pos, vel, mass = sc.big_scene(pick_ic(rng), nb, dom, rng, dev)
    p = torch.as_tensor(pos, device=dev)[None]
    v = torch.as_tensor(vel, device=dev)[None]
    m = torch.as_tensor(mass, device=dev)[None]
    k1 = torch.ones_like(m)
    phys = tt.Physics(grav=dict(eps=tt.EPS, g=1.0))
    F = args.big_steps
    r = tt.rollout(phys, p, v, m, k1, F, 0.1, 4, acc_fn=lambda a, b_, c, d: tt.grav_acc_large(a, c, d))
    P, V, A = r["pos"][0], r["vel"][0], r["acc_near"][0]
    half = sc.WINDOW / 2
    outs = dict(pos=[], vel=[], mask_f=[], acc_near=[], acc_far=[], acc_far_resid=[], a0=[], J=[], Q=[], exp_err=[], mass=[], centre=[], t0=[], n0=[])
    tries = 0
    while len(outs["pos"]) < args.windows and tries < args.windows * 20:
        tries += 1
        t0 = int(rng.integers(0, F - K + 1))
        c = P[t0, int(rng.integers(0, nb))] if rng.random() < 0.7 else torch.as_tensor(rng.uniform(-dom / 3, dom / 3, 2), device=dev)
        frames = slice(t0, t0 + K + 1)
        rel = P[frames] - c
        inw = (rel.abs().amax(-1) < half)
        ids = inw.any(0).nonzero()[:, 0]
        n = len(ids)
        if n < 8 or n > args.n_max:
            continue
        mk = inw[:, ids].to(torch.float64)
        mi = m[0, ids]
        xr = rel[:, ids]
        an = tt.grav_acc(xr, mi[None].expand(K + 1, -1), mk)
        af = A[frames][:, ids] - an
        a0, J, Q = tt.far_expansion(torch.zeros(K + 1, 2, device=dev, dtype=torch.float64), rel, m[0], (~inw).to(torch.float64))
        ex = tt.ExtField(a0, J, Q)
        err = ((ex.acc(xr) - af) ** 2).sum(-1).mul(mk).sum(1).sqrt() / ((af ** 2).sum(-1).mul(mk).sum(1).sqrt() + 1e-12)
        pad = args.n_max - n
        def pd(x, axis):
            sh_ = list(x.shape)
            sh_[axis] = pad
            return torch.cat([x, x.new_zeros(sh_)], axis)
        outs["pos"].append(pd(xr, 1)), outs["vel"].append(pd(V[frames][:, ids], 1)), outs["mask_f"].append(pd(mk, 1))
        outs["acc_near"].append(pd(an, 1)), outs["acc_far"].append(pd(af, 1)), outs["acc_far_resid"].append(pd((af - ex.acc(xr)) * mk[..., None], 1)), outs["mass"].append(pd(mi, 0))
        outs["a0"].append(a0), outs["J"].append(J), outs["Q"].append(Q), outs["exp_err"].append(err)
        outs["centre"].append(c), outs["t0"].append(torch.tensor(t0)), outs["n0"].append(inw[0].sum())
    return {k: torch.stack(v).float().cpu().numpy() for k, v in outs.items() if v}, tries


def gen_harvest(args, dev):
    os.makedirs(args.out, exist_ok=True)
    for sh in range(args.shards):
        if STOP:
            return
        tag = f"hv_s{args.seed}_{sh:03d}"
        path = os.path.join(args.out, tag + ".npz")
        if args.resume and os.path.exists(path):
            continue
        t0 = time.time()
        arrs, tries = harvest_one(args, dev, args.seed, sh)
        meta = dict(kind="harvest", seed=args.seed, shard=sh, steps=args.steps, git=git_rev(), n_big=args.n_big, domain=args.domain,
                    big_steps=args.big_steps, tries=tries, window=sc.WINDOW, dt=0.1, sub=4)
        save_atomic(path, meta=np.array(json.dumps(meta)), dt=np.float64(0.1), **arrs)
        print(f"{tag} windows={len(arrs['pos'])} mean_exp_err={arrs['exp_err'].mean():.3f} tries={tries} {time.time() - t0:.1f}s", flush=True)


def make_lower(spec, dev, span):
    if spec == "zero":
        return ti.ZeroLower()
    if spec.startswith("range"):
        return ti.RangeLower(float(spec[5:]))
    if spec.startswith("model:"):
        return ti.ModelLower(ti.load_scatter(spec[6:], dev), span)
    raise ValueError(spec)


def gen_tier1(args, dev):
    os.makedirs(args.out, exist_ok=True)
    lower = make_lower(args.lower, dev, args.span)
    for sh in range(args.shards):
        if STOP:
            return
        tag = f"t1_{lower.name.replace(':', '-').replace('/', '_')}_{args.states}_s{args.seed}_{sh:03d}"
        path = os.path.join(args.out, tag + ".npz")
        if args.resume and os.path.exists(path):
            continue
        t0 = time.time()
        rng = np.random.default_rng(seed_for(args.seed, "tier1", sh))
        n = int(np.exp(rng.uniform(np.log(args.n_range[0]), np.log(args.n_range[1]))))
        P, V, A_t, A_l, Ms, ics = [], [], [], [], [], []
        for _ in range(args.scenes):
            ic = str(rng.choice(args.ics.split(",")))
            pos, vel, mass = sc.big_scene(ic, n, args.span, rng, dev)
            p, v, m = (torch.as_tensor(x, device=dev)[None] for x in (pos, vel, mass))
            k1 = torch.ones_like(m)
            truth_acc = lambda a, b_, c, d: tt.grav_acc_large(a, c, d, direct_max=args.direct_max)
            if args.states == "truth":
                r = tt.rollout(None, p, v, m, k1, args.steps, 0.1, 4, acc_fn=truth_acc)
            else:
                r = tt.rollout(None, p, v, m, k1, args.steps, 0.1, 4, acc_fn=lambda a, b_, c, d: lower(a, b_, c, d))
            a_t = torch.stack([truth_acc(r["pos"][:, k], r["vel"][:, k], m, k1) for k in range(args.steps + 1)], 1)
            a_l = torch.stack([lower(r["pos"][:, k], r["vel"][:, k], m, k1) for k in range(args.steps + 1)], 1)
            P.append(r["pos"][0]), V.append(r["vel"][0]), A_t.append(a_t[0]), A_l.append(a_l[0]), Ms.append(m[0]), ics.append(ic)
        st = lambda L: torch.stack(L).float().cpu().numpy()
        meta = dict(kind="tier1", seed=args.seed, shard=sh, git=git_rev(), n=n, span=args.span, lower=lower.name, states=args.states,
                    ics=ics, truth="direct" if n <= args.direct_max else "cell-multipole", steps=args.steps, dt=0.1, sub=4)
        save_atomic(path, meta=np.array(json.dumps(meta)), pos=st(P), vel=st(V), mass=st(Ms), acc_truth=st(A_t), acc_lower=st(A_l),
                    resid=st([a - b for a, b in zip(A_t, A_l)]), dt=np.float64(0.1))
        print(f"{tag} N={n} scenes={args.scenes} truth={meta['truth']} {time.time() - t0:.1f}s", flush=True)


def truthcheck(args, dev):
    """Accuracy of the cell-multipole truth vs direct sum, and of the far-field expansion vs its direct-sum acceleration."""
    rng = np.random.default_rng(args.seed)
    res = {}
    for ic in ("plummer", "clumpy", "uniform", "disc"):
        pos, vel, mass = sc.big_scene(ic, args.check_n, 128.0, rng, dev)
        p, m = torch.as_tensor(pos, device=dev), torch.as_tensor(mass, device=dev)
        d = tt.grav_acc(p[None], m[None], torch.ones_like(m)[None])[0]
        c = tt.cell_grav_acc(p, m)
        e = (c - d).norm(dim=-1) / d.norm(dim=-1).clamp_min(1e-12)
        res[ic] = dict(median=float(e.median()), p99=float(e.quantile(0.99)), max=float(e.max()), rel_rms=float(((c - d) ** 2).sum().sqrt() / (d ** 2).sum().sqrt()))
        print("cell vs direct", ic, res[ic], flush=True)
    x = torch.randn(1, 5, 2, device=dev, dtype=torch.float64) * 3
    src = torch.randn(40, 2, device=dev, dtype=torch.float64) * 5 + torch.tensor([60.0, 10.0], device=dev, dtype=torch.float64)
    sm = torch.rand(40, device=dev, dtype=torch.float64) + 0.5
    a0, J, Q = tt.far_expansion(torch.zeros(1, 2, device=dev, dtype=torch.float64), src, sm, torch.ones(1, 40, device=dev, dtype=torch.float64))
    ex = tt.ExtField(a0, J, Q)
    d = src[None, None] - x[:, :, None]
    f = (d * ((d ** 2).sum(-1, keepdim=True) + tt.EPS ** 2) ** -1.5 * sm[None, None, :, None]).sum(2)
    res["expansion_rel_err"] = float(((ex.acc(x) - f) ** 2).sum().sqrt() / (f ** 2).sum().sqrt())
    print("far expansion rel err at |x|~4, src at ~60:", res["expansion_rel_err"], flush=True)
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tier0", "harvest", "tier1", "truthcheck"])
    ap.add_argument("--kinds", default="all")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--scenes", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="results/tiered/data")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--equil", type=int, default=200)
    ap.add_argument("--p-ext", type=float, default=0.8)
    ap.add_argument("--high-order", type=float, default=0.5)
    ap.add_argument("--p-rate", type=float, default=0.5)
    ap.add_argument("--noise-pos", type=float, default=0.0)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--n-big", type=int, default=8192)
    ap.add_argument("--domain", type=float, default=256.0)
    ap.add_argument("--big-steps", type=int, default=40)
    ap.add_argument("--span", type=float, default=128.0)
    ap.add_argument("--n-range", type=int, nargs=2, default=[512, 4096])
    ap.add_argument("--ics", default="plummer,clumpy,disc,uniform,cold_collapse,binary")
    ap.add_argument("--lower", default="range8")
    ap.add_argument("--states", choices=["truth", "lower"], default="truth")
    ap.add_argument("--direct-max", type=int, default=8192)
    ap.add_argument("--check-n", type=int, default=20000)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _sigterm)
    dev = torch.device(args.device)
    dict(tier0=gen_tier0, harvest=gen_harvest, tier1=gen_tier1, truthcheck=truthcheck)[args.mode](args, dev)


if __name__ == "__main__":
    main()
