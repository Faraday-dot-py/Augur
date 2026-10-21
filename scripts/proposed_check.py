"""Small-N check of the proposed scatter-field model (Exp E + R1b far kernel + ghost edge closure) against the exact relativistic softened all-pairs sim, BH collapse scene
(same IC as scatter_bh.py / scatter_bh_1b.py: Gaussian sigma 8 clipped +-29, vfac 0.3, c 10, dt .1, eps .5), mesh-only (pair term off, as in the 1e8 runs).

Per config (old / far / edge / new): err@k (mean position error per body vs exact), total relativistic energy E(t) = sum c^2 (gamma - 1) - pair potential, as |E_model(t) / E_truth(t) - 1|
(similarity to truth E(t), not conservation), core radius, |v|<c violations, non-finite count, fraction of bodies in the edge band (28 < max|x| < 32) and outside the box, model vs truth.

Usage: PYTHONPATH=. python scripts/proposed_check.py --ckpt E.pt --ghost-ckpt ghost_lsq.pt --kernel file:results/far_kernels/r1b_mlp.pt+cache --n 20000 --steps 200 --out results/proposed_check_n20000.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_bh as sb
from scripts import scatter_far
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

EPS = 0.5


@torch.no_grad()
def energy(pos, v, c, eps=EPS, elems=3.0e8):
    pos, v = pos.double(), v.double()
    gam = (1 - (v ** 2).sum(-1) / c ** 2).clamp(min=1e-12).rsqrt()
    ke = (c ** 2 * (gam - 1)).sum()
    n = pos.shape[0]
    ch = max(1, int(elems / n))
    pot = torch.zeros((), dtype=torch.float64, device=pos.device)
    for s in range(0, n, ch):
        e = min(n, s + ch)
        d = pos[None] - pos[s:e, None]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -0.5
        idx = torch.arange(s, e, device=pos.device)
        inv[idx - s, idx] = 0.0
        pot += inv.sum()
    return float(ke - 0.5 * pot)


def build(args, dev, variant, ckpt, kernel):
    m = tsf.build(tsf.EXPS["E"], variant, args.dt).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    m.eval()
    ops.no_pp(m)
    if kernel != "old":
        scatter_far.install_spec(m, kernel)
    return m


@torch.no_grad()
def rollout(model, pos0, vel0, steps, c, dt, dev):
    n = pos0.shape[0]
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
    field = model.init_field(1, dev)[0]
    P, V = [pos[0].cpu().numpy()], [sb.speed(mom, c)[0].cpu().numpy()]
    a, field = model.force(pos, sb.speed(mom, c), mass, mask, field)
    for _ in range(steps):
        mom = mom + 0.5 * dt * a
        pos = pos + dt * sb.speed(mom, c)
        a, field = model.force(pos, sb.speed(mom, c), mass, mask, field)
        mom = mom + 0.5 * dt * a
        P.append(pos[0].cpu().numpy())
        V.append(sb.speed(mom, c)[0].cpu().numpy())
    return np.stack(P), np.stack(V)


def stats(P, V, c):
    out = {"nonfinite": int((~np.isfinite(P).all(-1) | ~np.isfinite(V).all(-1)).sum()),
           "n_v_ge_c": int((np.linalg.norm(V, axis=-1) > c * (1 + 1e-6)).sum()),
           "vmax_over_c": float(np.nanmax(np.linalg.norm(V, axis=-1)) / c)}
    m = np.abs(P).max(-1)
    out["band_frac"] = ((m > 28) & (m < 32)).mean(-1).tolist()
    out["outside_frac"] = (m >= 32).mean(-1).tolist()
    out["core_r45"] = [sb.core_radius(f) for f in P]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ghost-ckpt", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seeds", type=lambda s: [int(i) for i in s.split(",")], default=[4738])
    ap.add_argument("--e-every", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    n, c, dt = args.n, args.c, args.dt
    cfgs = {"old": ("ms_kp_pot_v_g128", args.ckpt, "old"), "far": ("ms_kp_pot_v_g128", args.ckpt, args.kernel),
            "edge": ("ms_kp_pot_v_g128_ghost", args.ghost_ckpt, "old"), "new": ("ms_kp_pot_v_g128_ghost", args.ghost_ckpt, args.kernel)}
    res = {"args": vars(args), "seeds": {}}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos0 = np.clip(rng.normal(0.0, args.sigma, (n, 2)), -29.0, 29.0)
        vel0 = rng.normal(0.0, args.vfac * np.sqrt(n / (4 * args.sigma)), (n, 2))
        vel0 -= vel0.mean(0)
        Pt, Vt = gs.rollout_torch(pos0, vel0, args.steps, dev, dt=dt, substeps=4, eps=EPS, relativistic=True, c=c)
        Pt, Vt = np.asarray(Pt), np.asarray(Vt)
        ticks = list(range(0, args.steps + 1, args.e_every))
        Et = [energy(torch.tensor(Pt[t], device=dev), torch.tensor(Vt[t], device=dev), c) for t in ticks]
        r = {"truth": dict(stats(Pt, Vt, c), E=Et, ticks=ticks)}
        for name, (variant, ck, kern) in cfgs.items():
            m = build(args, dev, variant, ck, kern)
            P, V = rollout(m, pos0, vel0, args.steps, c, dt, dev)
            E = [energy(torch.tensor(P[t], device=dev), torch.tensor(V[t], device=dev), c) for t in ticks]
            err = np.linalg.norm(P - Pt, axis=-1).mean(1)
            d = stats(P, V, c)
            d.update(err=err.tolist(), E=E, dE_vs_truth=[abs(e / et - 1) for e, et in zip(E, Et)])
            r[name] = d
            print(f"seed {seed} {name}: err@20/50/100/{args.steps} {[round(float(err[i]), 3) for i in (20, 50, 100, args.steps) if i < len(err)]} "
                  f"dE_vs_truth@{args.steps} {d['dE_vs_truth'][-1]:.4f} nonfinite {d['nonfinite']} n_v>=c {d['n_v_ge_c']} band_max {max(d['band_frac']):.4f} "
                  f"(truth {max(r['truth']['band_frac']):.4f}) outside_end {d['outside_frac'][-1]:.3f} (truth {r['truth']['outside_frac'][-1]:.3f})", flush=True)
            del m
        res["seeds"][str(seed)] = r
        json.dump(res, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
