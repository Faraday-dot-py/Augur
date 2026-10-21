"""Evaluation of far-kernel variants on the wide tiled scene (docs/debugging/scatter-far-kernel-fix.md).

Wide scene = 16 tiles of 64 (extent 256), N 4800, sigma 32, warm, tiles halo 16 (G=192 copies), vs exact fp64 all-pairs.
Phases:
  table    kernel force ratio vs the true softened force (r bins), sign / monotone / r<=32 checks vs the old kernel
  wide     err@5/10/20 vs exact, difference to the single G=512 grid, tick-0 force error, momentum drift; per kernel spec and far_c
  farerr   far-term decomposition at tick 0: tiled far acceleration of own bodies vs exact acceleration from mass outside the tile box
Usage: PYTHONPATH=. python scripts/scatter_far_eval.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --phase wide --kernels old,oracle64 --out results/x.json
"""
import argparse
import json
import time
import types

import torch

from scripts import scatter_far as sfar
from scripts import scatter_scaling_ops as ops
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling import TiledScatter
from scripts.scatter_tiling_eval import DT, EPS, SUB, load

L, N, SIGMA = 256.0, 4800, 32.0
NT, KIND = 4, "gauss"
SEEDS = (4738, 9000, 12000)
HELD = (9000, 12000)


def set_scene(nt, n, sigma, kind):
    global L, N, SIGMA, NT, KIND
    NT, N, SIGMA, KIND, L = nt, n, sigma, kind, 64.0 * nt


def scene(seed):
    if KIND == "gauss":
        return make_scene(N, SIGMA, 1.0, L, seed, 0)
    if KIND == "two":
        a = make_scene(N // 2, SIGMA, 1.0, L, seed, 0)
        b = make_scene(N - N // 2, SIGMA, 1.0, L, seed, 1)
        p = torch.cat([torch.tensor(a[0]) + torch.tensor([-L / 4, 0.0]), torch.tensor(b[0]) + torch.tensor([L / 4, 0.0])]).numpy()
        v = torch.cat([torch.tensor(a[1]), torch.tensor(b[1])])
        return p, (v - v.mean(0)).numpy()
    if KIND == "uniform":
        import numpy as np
        rng = np.random.default_rng([seed, N, 7])
        p = rng.uniform(-0.45 * L, 0.45 * L, (N, 2))
        v = rng.normal(0.0, np.sqrt(N / (4 * 0.25 * L)), (N, 2))
        return p, v - v.mean(0)
    raise ValueError(KIND)


def rel(a, ref):
    return float((a - ref).norm(dim=-1).sum() / ref.norm(dim=-1).sum())


@torch.no_grad()
def run_single(m, pos, vel, mass, steps):
    ops.retarget(m, round(L / 0.5), L)
    msk = torch.ones(1, pos.shape[0], device=pos.device)
    st = (m.init_field(1, pos.device)[0], None)
    a0, _ = m.force(pos[None], vel[None], mass[None], msk, st[0])
    p, v = pos[None], vel[None]
    Ps = [p[0]]
    for _ in range(steps):
        p, v, st, _ = m.step(p, v, mass[None], msk, st)
        Ps.append(p[0])
    return a0[0], torch.stack(Ps).double(), v[0]


@torch.no_grad()
def run_tiled(T, pos, vel, mass, steps):
    a0, _ = T.force(pos, vel, mass, T.init_field(pos.device))
    ts = (T.init_field(pos.device), None)
    p, v = pos, vel
    Pl = [p]
    for _ in range(steps):
        p, v, ts = T.step(p, v, mass, ts)
        Pl.append(p)
    return a0, torch.stack(Pl).double(), v


def exact_far_field(T, pos, mass, bs=4096):
    """(T,2,G,G) exact softened acceleration at every fine cell centre of each tile's extended box from the bodies OUTSIDE that box."""
    n, G, h, E = T.nt, T.G, T.h, T.E
    cen = (torch.arange(G, device=pos.device, dtype=torch.float64) + 0.5) * h - E / 2
    out = []
    for k in range(n * n):
        tx, ty = divmod(k, n)
        c = torch.tensor([T.X0 + (tx + 0.5) * T.Lt, T.X0 + (ty + 0.5) * T.Lt], device=pos.device, dtype=torch.float64)
        rel_ = pos.double() - c
        outside = (rel_.abs() >= E / 2).any(-1)
        src, ms = rel_[outside], mass.double()[outside]
        gy, gx = torch.meshgrid(cen, cen, indexing="ij")
        cells = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
        acc = torch.empty_like(cells)
        for s in range(0, cells.shape[0], bs):
            d = src[None, :, :] - cells[s:s + bs, None, :]
            w = ms[None, :] * ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
            acc[s:s + bs] = (d * w[..., None]).sum(1)
        out.append(acc.reshape(G, G, 2).permute(2, 0, 1))
    return torch.stack(out).float()


def make_exactfar(T):
    cur = {}
    orig = T.force

    def force(pos, vel, mass, field, tiles=None):
        cur["pm"] = (pos, mass)
        return orig(pos, vel, mass, field, tiles)

    def far_acc(rho):
        return exact_far_field(T, *cur["pm"])
    T.force = force
    T._far_acc = far_acc
    return T


@torch.no_grad()
def phase_table(m, dev, specs, out):
    old = sfar.old_fn(m)
    rs = torch.tensor([1, 2, 4, 8, 16, 32, 40, 48, 64, 80, 100, 150, 200, 256, 300, 400, 512, 640, 800, 1000, 1500], device=dev, dtype=torch.float32)
    res = {}
    for spec in specs:
        kf = sfar.kfun_from_spec(m, spec) or sfar.old_fn(m)
        r = torch.logspace(0, 3.2, 6000, device=dev)
        F = sfar.force_of(kf, r)
        Fo = sfar.force_of(old, r)
        ratio = F / sfar.f_true(r)
        sel = (r >= 4) & (r <= 512)
        selw = (r >= 4) & (r <= 1500)
        sel32 = (r >= 4) & (r <= 32)
        sel_far = r >= 32
        dec = F[sel_far][1:] - F[sel_far][:-1]
        relinc = (dec / F[sel_far][:-1].abs().clamp_min(1e-12))
        wi = int(relinc.argmax())
        mid = r[sel_far][1:] <= 512
        mono512 = bool((dec[mid] <= 1e-12).all())
        row = {"ratio_at": {str(int(x)): float(sfar.force_of(kf, x[None])[0] / sfar.f_true(x)) for x in rs},
               "ratio_min_4_512": float(ratio[sel].min()), "ratio_max_4_512": float(ratio[sel].max()),
               "max_abs_dev_4_512": float((ratio[sel] - 1).abs().max()),
               "ratio_min_4_1500": float(ratio[selw].min()), "ratio_max_4_1500": float(ratio[selw].max()),
               "min_force_ge_4": float(F[r >= 4].min()), "monotone_decay_ge32": bool((dec <= 1e-12).all()),
               "monotone_decay_32_512": mono512, "worst_force_increase_rel_per_step": float(relinc.max()), "worst_increase_at_r": float(r[sel_far][wi]),
               "max_rel_dev_vs_old_4_32": float(((F[sel32] - Fo[sel32]).abs() / Fo[sel32].abs()).max()),
               "max_rel_dev_vs_old_1_32": float(((F[r <= 32] - Fo[r <= 32]).abs() / Fo[r <= 32].abs().clamp_min(1e-3)).max())}
        res[spec] = row
        print(spec, json.dumps(row), flush=True)
        json.dump(res, open(out, "w"))


def phase_wide(m, dev, specs, fars, out, variants, seeds=SEEDS, steps=20):
    rows = []
    ks = [k for k in (5, 10, 20, 50, 100) if k <= steps]
    for seed in seeds:
        p0, v0 = scene(seed)
        Pt, _ = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), steps, DT, SUB, EPS)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)
        mass = torch.ones(N, device=dev)
        ax = ops.exact_acc(pos.double(), EPS).float()
        for spec in specs:
            sfar.install_spec(m, spec)
            a_s, P_s, v_s = run_single(m, pos, vel, mass, steps)
            base = {"seed": seed, "spec": spec}
            r = {**base, "variant": "single", "far_c": None, "force_vs_exact": rel(a_s, ax),
                 "err": {str(k): float((P_s[k] - Pt[k]).norm(dim=-1).mean()) for k in ks}}
            rows.append(r)
            print(json.dumps(r), flush=True)
            for var in variants:
                for c in (fars if var != "halo" else [None]):
                    T = TiledScatter(m, NT, 64, 16, ghosts=True, field_x=True, far=var != "halo", far_c=c or 4)
                    if var == "exactfar":
                        make_exactfar(T)
                    a_t, P_t, v_t = run_tiled(T, pos, vel, mass, steps)
                    ts = (T.init_field(dev), None)
                    for _ in range(2):
                        T.step(pos, vel, mass, ts)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(5):
                        T.step(pos, vel, mass, ts)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) * 200
                    r = {**base, "ms_tick": ms, "variant": var, "far_c": c, "force_vs_exact": rel(a_t, ax), "force_vs_single": rel(a_t, a_s),
                         "err": {str(k): float((P_t[k] - Pt[k]).norm(dim=-1).mean()) for k in ks},
                         "diff_single20": float((P_t[ks[2]] - P_s[ks[2]]).norm(dim=-1).mean()),
                         "dP": float((v_t.sum(0).norm() - vel.sum(0).norm()).abs()), "finite": bool(torch.isfinite(P_t).all())}
                    rows.append(r)
                    print(json.dumps(r), flush=True)
            sfar.install(m, None)
        json.dump(rows, open(out, "w"))


@torch.no_grad()
def phase_farerr(m, dev, specs, fars, out, seeds=SEEDS):
    rows = []
    for seed in seeds:
        p0, v0 = scene(seed)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)
        mass = torch.ones(N, device=dev)
        a_tot = ops.exact_acc(pos.double(), EPS).float()
        ops.retarget(m, round(L / 0.5), L)
        for spec in specs:
            sfar.install_spec(m, spec)
            for c in fars:
                T = TiledScatter(m, NT, 64, 16, ghosts=True, field_x=True, far=True, far_c=c)
                ef = exact_far_field(T, pos, mass)
                old = T._scope()
                try:
                    P, V, M, K, OWN, INS, idx = T._pack(pos, vel, mass, list(range(16)))
                    x = m.scatter(P, V, M, K * INS)
                    far = T._far_acc(x[:, :1])
                    a_far = m.gather(far, P)
                    a_ex = m.gather(ef, P)
                finally:
                    T._restore(old)
                num = den = tot = 0.0
                bins = {"16-24": [0., 0.], "24-40": [0., 0.], "40+": [0., 0.]}
                for j, i in enumerate(idx):
                    no = int(OWN[j].sum())
                    sel = i[:no]
                    d = (a_far[j, :no] - a_ex[j, :no]).norm(dim=-1)
                    e = a_ex[j, :no].norm(dim=-1)
                    t = a_tot[sel].norm(dim=-1)
                    num += float(d.sum())
                    den += float(e.sum())
                    tot += float(t.sum())
                    rel_ = P[j, :no].abs().amax(-1)
                    dedge = T.E / 2 - rel_
                    for nm, mk in (("16-24", dedge < 24), ("24-40", (dedge >= 24) & (dedge < 40)), ("40+", dedge >= 40)):
                        bins[nm][0] += float(d[mk].sum())
                        bins[nm][1] += float(e[mk].sum())
                row = {"seed": seed, "spec": spec, "far_c": c, "far_rel_err": num / den, "far_err_over_total_acc": num / tot,
                       "far_over_total_acc": den / tot, "by_edge_dist": {k: v[0] / max(v[1], 1e-12) for k, v in bins.items()}}
                rows.append(row)
                print(json.dumps(row), flush=True)
            sfar.install(m, None)
        json.dump(rows, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phase", choices=["table", "wide", "farerr"], required=True)
    ap.add_argument("--kernels", default="old")
    ap.add_argument("--far-c", default="4")
    ap.add_argument("--variants", default="far")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nt", type=int, default=4, help="tiles per side (domain = 64 * nt)")
    ap.add_argument("--n", type=int, default=4800)
    ap.add_argument("--sigma", type=float, default=32.0)
    ap.add_argument("--kind", default="gauss", choices=["gauss", "two", "uniform"])
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    set_scene(a.nt, a.n, a.sigma, a.kind)
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    m = load(a.ckpt, dev)
    specs = a.kernels.split(",")
    fars = [int(x) for x in a.far_c.split(",")]
    seeds = tuple(int(x) for x in a.seeds.split(","))
    if a.phase == "table":
        phase_table(m, dev, specs, a.out)
    elif a.phase == "wide":
        phase_wide(m, dev, specs, fars, a.out, a.variants.split(","), seeds, a.steps)
    else:
        phase_farerr(m, dev, specs, fars, a.out, seeds)


if __name__ == "__main__":
    main()
