"""Axisymmetric mean-field reference for the BH-collapse scene: no mesh, no shot noise from pair forces, same softened law, same relativistic integrator.

M representative bodies (weight N/M each) sampled from the scene IC; the force on a body at radius r is the azimuthally averaged softened attraction of all mass rings
(table T[i, j] = mean over phi of the radial component of d / (|d|^2 + eps^2)^1.5 from a ring of radius r_j at a point at radius r_i), so it is purely radial and L is
conserved exactly. N enters only through the weights, so the limit N -> infinity (collisionless, cold in p/N) can be run at any N. Planar positions with the 3-D-form
force law of the project (no shell theorem: the ring table replaces it). Integrator: relativistic kick-drift-kick as gravity_sim.rollout_torch (--sub substeps per tick).

Writes <out>.npz with per-tick radial counts on the same bins as bh_wave_run (RLIN, RCORE), L, KE.

Usage: python scripts/bh_shell_ref.py --n 1e8 --m 20000 --sub 4 --steps 150 --out results/bhwave/ref_n1e8
"""
import argparse
import time

import numpy as np

EPS = 0.5
RLIN = np.arange(0, 48.01, 0.25)
RCORE = np.arange(0, 4.001, 0.02)


def nodes():
    return np.unique(np.concatenate([np.arange(0, 12, 0.1), np.arange(12, 40, 0.25), np.arange(40, 260.01, 1.0)]))


def ring_table(r, nphi=192):
    phi = (np.arange(nphi) + 0.5) * np.pi / nphi
    cp = np.cos(phi)
    T = np.zeros((len(r), len(r)))
    for i, ri in enumerate(r):
        rj = r[None, :, None]
        d2 = ri ** 2 + rj ** 2 - 2 * ri * rj * cp[None, None, :] + EPS ** 2
        T[i] = ((rj * cp - ri) / d2 ** 1.5).mean(-1)[0]
    return T


def split(rp, rn):
    k = np.clip(np.searchsorted(rn, rp) - 1, 0, len(rn) - 2)
    f = np.clip((rp - rn[k]) / (rn[k + 1] - rn[k]), 0, 1)
    return k, f


class Ref:
    def __init__(self, n, c):
        self.rn = nodes()
        self.T = ring_table(self.rn)
        self.n, self.c = n, c

    def acc(self, x, w):
        r = np.linalg.norm(x, axis=1)
        k, f = split(r, self.rn)
        m = np.bincount(k, weights=w * (1 - f), minlength=len(self.rn)) + np.bincount(k + 1, weights=w * f, minlength=len(self.rn) + 1)[: len(self.rn)]
        F = self.T @ m
        g = F.copy()
        g[1:] = F[1:] / self.rn[1:]
        g[0] = g[1]
        gp = g[k] * (1 - f) + g[k + 1] * f
        return gp[:, None] * x


def speed(p, c):
    return p / np.sqrt(1 + (p ** 2).sum(-1, keepdims=True) / c ** 2)


def diag(x, p, c, wt):
    r = np.linalg.norm(x, axis=1)
    v = speed(p, c)
    out = {}
    out["rad_lin"] = np.histogram(r, RLIN, weights=wt)[0]
    out["rad_core"] = np.histogram(r, RCORE, weights=wt)[0]
    rr = np.maximum(r, 1e-9)
    out["vr_lin"] = np.histogram(r, RLIN, weights=wt * (v * x).sum(1) / rr)[0]
    lz = x[:, 0] * p[:, 1] - x[:, 1] * p[:, 0]
    out["L"] = float((wt * lz).sum())
    out["absl"] = float((wt * np.abs(lz)).sum())
    gam = 1 / np.sqrt(np.clip(1 - (v ** 2).sum(1) / c ** 2, 1e-12, None))
    out["KE"] = float((wt * c ** 2 * (gam - 1)).sum())
    out["pv"] = float((wt * (p * v).sum(1)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, required=True)
    ap.add_argument("--m", type=int, default=20000)
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n, m, c, dt = args.n, args.m, args.c, args.dt
    rng = np.random.default_rng(args.seed)
    x = np.clip(rng.normal(0, args.sigma, (m, 2)), -29.0, 29.0)
    v = rng.normal(0, args.vfac * np.sqrt(n / (4 * args.sigma)), (m, 2))
    if args.rot:
        v = v + args.rot * np.stack([-x[:, 1], x[:, 0]], 1)
    v -= v.mean(0)
    x -= x.mean(0)
    vm = np.linalg.norm(v, axis=1, keepdims=True).clip(max=0.99 * c)
    p = v / np.sqrt(np.clip(1 - (vm / c) ** 2, 1e-6, None))
    w = np.full(m, n / m)
    wt = np.full(m, n / m)
    ref = Ref(n, c)
    h = dt / args.sub
    a = ref.acc(x, w)
    rows = [diag(x, p, c, wt)]
    t0 = time.perf_counter()
    for t in range(args.steps):
        for _ in range(args.sub):
            p = p + 0.5 * h * a
            x = x + h * speed(p, c)
            a = ref.acc(x, w)
            p = p + 0.5 * h * a
        rows.append(diag(x, p, c, wt))
    res = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}
    np.savez_compressed(args.out + ".npz", **res)
    print(f"n={n:g} m={m} sub={args.sub} wall {time.perf_counter() - t0:.0f}s L0 {res['L'][0]:.3e} L_end {res['L'][-1]:.3e}", flush=True)


if __name__ == "__main__":
    main()
