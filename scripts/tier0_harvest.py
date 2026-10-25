"""Harvest 64x64 windows from large softened-gravity rollouts with the direct-sum far-field Taylor expansion to THIRD order
(a0, J, Q, C of the acceleration about the window centre; sources outside the window). Same windows as tiered_gen.py harvest, plus C and
the relative far-force residual for order 2 and order 3. Needs tiered_gen / tiered_scenes / tiered_truth on the path.

Usage: PYTHONPATH=. python scripts/tier0_harvest.py --shards 120 --windows 32 --steps 24 --n-max 300 --seed 4738 --out data3
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import tiered_gen as tg
from scripts import tiered_scenes as sc
from scripts import tiered_truth as tt


def third(d, eps):
    """f, df, ddf, dddf of f(d) = d (|d|^2+eps^2)^-1.5 for d (M,2); dddf (M,2,2,2,2)."""
    fn = lambda x: tt._fkl(x, eps)[2]
    return torch.func.vmap(torch.func.jacfwd(fn))(d)


def far_third(center, src_pos, src_mass, src_ok, eps=tt.EPS):
    """center (T,2) per-frame centre of one window, src_pos (T,M,2), weights src_mass*src_ok (T,M) -> a0 (T,2), J, Q, C."""
    w = src_mass * src_ok
    d = src_pos - center[:, None, :]
    f, df, ddf = tt._fkl(d, eps)
    dddf = torch.stack([third(d[t], eps) for t in range(d.shape[0])])
    a0 = (w[..., None] * f).sum(1)
    J = -(w[..., None, None] * df).sum(1)
    Q = (w[..., None, None, None] * ddf).sum(1)
    C = -(w[..., None, None, None, None] * dddf).sum(1)
    return a0, J, Q, C


def acc3(a0, J, Q, C, x):
    return (a0[:, None, :] + torch.einsum("bij,bnj->bni", J, x) + 0.5 * torch.einsum("bijk,bnj,bnk->bni", Q, x, x)
            + torch.einsum("bijkl,bnj,bnk,bnl->bni", C, x, x, x) / 6)


def harvest_one(args, dev, seed, sh):
    rng = np.random.default_rng(tg.seed_for(seed, "harvest3", sh))
    dom, nb, K = args.domain, args.n_big, args.steps
    pos, vel, mass = sc.big_scene(tg.pick_ic(rng), nb, dom, rng, dev)
    p, v, m = (torch.as_tensor(x, device=dev)[None] for x in (pos, vel, mass))
    phys = tt.Physics(grav=dict(eps=tt.EPS, g=1.0))
    F = args.big_steps
    r = tt.rollout(phys, p, v, m, torch.ones_like(m), F, 0.1, 4, acc_fn=lambda a, b_, c, d: tt.grav_acc_large(a, c, d))
    P, V, A = r["pos"][0], r["vel"][0], r["acc_near"][0]
    half = sc.WINDOW / 2
    keys = ("pos", "vel", "mask_f", "acc_far", "a0", "J", "Q", "C", "exp_err2", "exp_err3", "mass", "centre")
    outs = {k: [] for k in keys}
    tries = 0
    while len(outs["pos"]) < args.windows and tries < args.windows * 20:
        tries += 1
        t0 = int(rng.integers(0, F - K + 1))
        c = P[t0, int(rng.integers(0, nb))] if rng.random() < 0.7 else torch.as_tensor(rng.uniform(-dom / 3, dom / 3, 2), device=dev)
        frames = slice(t0, t0 + K + 1)
        rel = P[frames] - c
        inw = rel.abs().amax(-1) < half
        ids = inw.any(0).nonzero()[:, 0]
        n = len(ids)
        if n < 8 or n > args.n_max:
            continue
        mk = inw[:, ids].to(torch.float64)
        mi = m[0, ids]
        xr = rel[:, ids]
        an = tt.grav_acc(xr, mi[None].expand(K + 1, -1), mk)
        af = A[frames][:, ids] - an
        a0, J, Q, C = far_third(torch.zeros(K + 1, 2, device=dev, dtype=torch.float64), rel, m[0][None].expand(K + 1, -1), (~inw).to(torch.float64))
        den = ((af ** 2).sum(-1).mul(mk).sum(1).sqrt() + 1e-12)
        e2 = ((acc3(a0, J, Q, torch.zeros_like(C), xr) - af) ** 2).sum(-1).mul(mk).sum(1).sqrt() / den
        e3 = ((acc3(a0, J, Q, C, xr) - af) ** 2).sum(-1).mul(mk).sum(1).sqrt() / den
        pad = args.n_max - n
        pd = lambda x, ax: torch.cat([x, x.new_zeros(*x.shape[:ax], pad, *x.shape[ax + 1:])], ax)
        outs["pos"].append(pd(xr, 1)), outs["vel"].append(pd(V[frames][:, ids], 1)), outs["mask_f"].append(pd(mk, 1))
        outs["acc_far"].append(pd(af, 1)), outs["mass"].append(pd(mi, 0))
        for k, x in (("a0", a0), ("J", J), ("Q", Q), ("C", C), ("exp_err2", e2), ("exp_err3", e3), ("centre", c)):
            outs[k].append(x)
    return {k: torch.stack(v).float().cpu().numpy() for k, v in outs.items() if v}, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=120)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--n-max", type=int, default=300)
    ap.add_argument("--n-big", type=int, default=8192)
    ap.add_argument("--domain", type=float, default=256.0)
    ap.add_argument("--big-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="data3")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    for sh in range(a.shards):
        path = os.path.join(a.out, f"hv3_s{a.seed}_{sh:03d}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        arrs, tries = harvest_one(a, dev, a.seed, sh)
        if not arrs:
            continue
        tg.save_atomic(path, meta=np.array(json.dumps(dict(shard=sh, seed=a.seed, steps=a.steps, tries=tries, order=3))), **arrs)
        print(f"hv3 {sh} windows={len(arrs['pos'])} err2={arrs['exp_err2'].mean():.3f} err3={arrs['exp_err3'].mean():.3f} tries={tries} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
