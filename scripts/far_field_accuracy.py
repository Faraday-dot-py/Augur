"""Rollout accuracy vs time of the central-force model with and without the
far field, against the exact all-pairs analytic truth on one uniform
scaled-init cloud. Variants: model near-only (radius 4), model truncated at
radius 100 (the earlier 1000-body configuration), model near+mesh far field,
and the analytic force with near+mesh (isolates mesh error from model error).

Usage: PYTHONPATH=. python3 scripts/far_field_accuracy.py --bodies 10000 --steps 2000 --out results/far_field_accuracy.npz
"""
import argparse

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from scripts.far_field_check import exact_accel
from scripts.gravity_1b import analytic_force, far_accel, init_state, model_force, tiled_accel


@torch.no_grad()
def energy(pos, vel, eps, chunk=2000):
    pe = 0.0
    for i in range(0, pos.shape[0], chunk):
        rel = pos[None, :, :] - pos[i:i + chunk, None, :]
        r = torch.sqrt((rel ** 2).sum(-1) + eps ** 2)
        pe = pe - (1.0 / r).double().sum()
    return float(0.5 * (vel.double() ** 2).sum() + 0.5 * (pe + pos.shape[0] / eps))


def make_accel(kind, force, radius, grid):
    if kind == "exact":
        return lambda p: exact_accel(force, p)
    if kind == "trunc":
        return lambda p: tiled_accel(force, p, radius, 4)
    return lambda p: tiled_accel(force, p, radius, 4) + far_accel(force, p, radius, grid)


@torch.no_grad()
def rollout(accel, pos, vel, steps, dt, record, eps):
    out = {"gap": [], "energy": [], "mom": [], "step": []}
    a = accel(pos)
    for k in range(steps + 1):
        if k % record == 0:
            out["step"].append(k)
            out["energy"].append(energy(pos, vel, eps))
            out["mom"].append(float(vel.double().sum(0).norm()))
            out["pos"] = out.get("pos", []) + [pos.clone()]
        if k == steps:
            break
        pos = pos + vel * dt + 0.5 * dt * dt * a
        a1 = accel(pos)
        vel = vel + 0.5 * dt * (a + a1)
        a = a1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--bodies", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--record", type=int, default=20)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="results/far_field_accuracy.npz")
    args = ap.parse_args()
    dev = torch.device("cuda")
    dyn = CentralForceDynamics(dt=args.dt).to(dev)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    mf, af = model_force(dyn), analytic_force(args.eps)
    pos0, vel0, spread = init_state(args.bodies, dev, args.seed)
    variants = {
        "truth (exact all-pairs)": ("exact", af, 0),
        "model, cutoff 4": ("trunc", mf, 4.0),
        "model, cutoff 100": ("trunc", mf, 100.0),
        "model, cutoff 4 + far field": ("far", mf, 4.0),
        "analytic, cutoff 4 + far field": ("far", af, 4.0),
    }
    res = {}
    for name, (kind, f, r) in variants.items():
        res[name] = rollout(make_accel(kind, f, r, args.grid), pos0.clone(), vel0.clone(), args.steps, args.dt, args.record, args.eps)
        print(name, "done", flush=True)
    truth = res["truth (exact all-pairs)"]
    save = {"step": np.array(truth["step"]), "dt": args.dt, "bodies": args.bodies, "spread": spread}
    for i, name in enumerate(res):
        gap = [float((p - t).norm(dim=1).mean()) for p, t in zip(res[name]["pos"], truth["pos"])]
        save[f"gap_{i}"] = np.array(gap)
        save[f"energy_{i}"] = np.array(res[name]["energy"])
        save[f"mom_{i}"] = np.array(res[name]["mom"])
        save[f"name_{i}"] = name
    np.savez(args.out, **save)


if __name__ == "__main__":
    main()
