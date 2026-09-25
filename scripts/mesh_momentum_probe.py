"""Momentum of the particle-mesh far field. Static probe: total mesh force
sum_i a_i on a uniform cloud for each deposit/gather mode, N and grid. Rollout:
analytic near+mesh variants vs exact all-pairs truth (|P|, energy, gap). Timing
at large N.

Usage: PYTHONPATH=. python3 scripts/mesh_momentum_probe.py --out results/mesh_momentum_probe.npz
"""
import argparse
import time

import numpy as np
import torch

from scripts.far_field_accuracy import energy, rollout
from scripts.far_field_check import exact_accel
from scripts.gravity_1b import analytic_force, init_state, tiled_accel
from scripts.gravity_1b import far_accel as _far_accel


def far_accel(force, pos, radius, grid, mode):
    if mode.endswith("64"):
        return _far_accel(force, pos, radius, grid, mode=mode[:-2], fft64=True)
    return _far_accel(force, pos, radius, grid, mode=mode, fft64=False)


def mesh_variant(force, radius, grid, mode, meansub):
    def far(p):
        a = far_accel(force, p, radius, grid, mode=mode)
        return a - a.mean(0, keepdim=True) if meansub else a
    return far


@torch.no_grad()
def static(dev, force, seed, radius):
    print("static: N grid mode |sum a_mesh|/(N mean|a|)  |sum a_near+mesh|  rel err vs exact", flush=True)
    for n in (2000, 10000, 20000):
        pos, vel, spread = init_state(n, dev, seed)
        ex = exact_accel(force, pos)
        norm = float(ex.norm(dim=1).mean())
        near = tiled_accel(force, pos, radius, 4)
        print(f"N {n} |sum near| {float(near.double().sum(0).norm()):.3g} |sum exact| {float(ex.double().sum(0).norm()):.3g} mean|a| {norm:.4g}", flush=True)
        for grid in (64, 128, 256, 512):
            for mode in ("ngp-bilinear", "cic", "cic64", "ngp"):
                m = far_accel(force, pos, radius, grid, mode=mode)
                s = float(m.double().sum(0).norm())
                tot = float((near + m).double().sum(0).norm())
                e = float((near + m - ex).norm(dim=1).mean() / norm)
                em = float((near + m - m.mean(0, keepdim=True) - ex).norm(dim=1).mean() / norm)
                print(f"  {n} {grid} {mode:13s} mesh {s / (n * norm):.3e} ({s:.3g})  total {tot:.3g}  err {e:.4f}  err_meansub {em:.4f}", flush=True)


def rollouts(dev, force, args):
    pos0, vel0, spread = init_state(args.bodies, dev, args.seed)
    variants = {
        "truth": lambda p: exact_accel(force, p),
        "ngp-bilinear (current)": ("ngp-bilinear", False),
        "cic": ("cic", False),
        "cic fft64": ("cic64", False),
        "ngp": ("ngp", False),
        "ngp-bilinear + meansub": ("ngp-bilinear", True),
        "cic + meansub": ("cic", True),
        "cic fft64 + meansub": ("cic64", True),
    }
    res = {}
    for name, v in variants.items():
        if callable(v):
            acc = v
        else:
            far = mesh_variant(force, 4.0, args.grid, *v)
            acc = lambda p, far=far: tiled_accel(force, p, 4.0, 4) + far(p)
        res[name] = rollout(acc, pos0.clone(), vel0.clone(), args.steps, args.dt, args.record, args.eps)
        print(name, "done", flush=True)
    truth = res["truth"]
    save = {"step": np.array(truth["step"])}
    for i, name in enumerate(res):
        gap = [float((p - t).norm(dim=1).mean()) for p, t in zip(res[name]["pos"], truth["pos"])]
        save[f"gap_{i}"] = np.array(gap)
        save[f"energy_{i}"] = np.array(res[name]["energy"])
        save[f"mom_{i}"] = np.array(res[name]["mom"])
        save[f"name_{i}"] = name
        k = [j for j, s in enumerate(truth["step"]) if s in (0, 20, 200, 1000, args.steps)]
        print(f"{name:26s} " + "  ".join(f"t{truth['step'][j]}: P {res[name]['mom'][j]:.3g} dE/E0 {(res[name]['energy'][j] - truth['energy'][j]) / abs(truth['energy'][0]):+.3f} gap {gap[j]:.3g}" for j in k), flush=True)
    return save


@torch.no_grad()
def timing(dev, force, seed, n, grid, steps):
    pos, vel, spread = init_state(n, dev, seed)
    for mode in ("ngp-bilinear", "cic", "cic64"):
        far_accel(force, pos, 4.0, grid, mode=mode)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(steps):
            far_accel(force, pos, 4.0, grid, mode=mode)
        torch.cuda.synchronize()
        a = far_accel(force, pos, 4.0, grid, mode=mode)
        print(f"timing N {n} grid {grid} {mode}: {(time.time() - t) / steps:.3f} s per mesh call, |sum a| {float(a.double().sum(0).norm()):.3g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--record", type=int, default=20)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--time-bodies", type=int, default=1000000)
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--out", default="results/mesh_momentum_probe.npz")
    args = ap.parse_args()
    dev = torch.device("cuda")
    force = analytic_force(args.eps)
    if not args.skip_static:
        static(dev, force, args.seed, 4.0)
    timing(dev, force, args.seed, args.time_bodies, 1024, 5)
    np.savez(args.out, **rollouts(dev, force, args))


if __name__ == "__main__":
    main()
