"""Two-cluster collision: exact all-pairs analytic truth vs CentralForceDynamics
(cutoff 4 + mesh far field) from the same initial state, per-body positions
recorded for rendering. Each cluster is a Gaussian blob scaled to virial
equilibrium (KE = |PE| / 2), given a bulk velocity toward the other with an
impact parameter.

Usage: PYTHONPATH=. python3 scripts/gravity_collision.py --bodies 10000 --out results/collision_10k.npz
"""
import argparse

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from scripts.far_field_accuracy import energy, make_accel
from scripts.gravity_1b import analytic_force, model_force


def init_cluster(n, centre, sigma, eps, gen, dev):
    pos = torch.randn(n, 2, generator=gen, device=dev) * sigma + centre
    vel = torch.randn(n, 2, generator=gen, device=dev)
    vel -= vel.mean(0)
    pe = energy(pos, torch.zeros_like(pos), eps)
    ke = 0.5 * float((vel.double() ** 2).sum())
    return pos, vel * (0.5 * abs(pe) / ke) ** 0.5


@torch.no_grad()
def rollout(accel, pos, vel, steps, dt, record):
    frames = [pos.cpu().numpy()]
    a = accel(pos)
    for k in range(1, steps + 1):
        pos = pos + vel * dt + 0.5 * dt * dt * a
        a1 = accel(pos)
        vel = vel + 0.5 * dt * (a + a1)
        a = a1
        if k % record == 0:
            frames.append(pos.cpu().numpy())
    return np.stack(frames), pos, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--bodies", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--record", type=int, default=5)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=30.0)
    ap.add_argument("--separation", type=float, default=250.0)
    ap.add_argument("--impact", type=float, default=60.0)
    ap.add_argument("--approach", type=float, default=3.0, help="each cluster's bulk speed toward the other")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="results/collision_10k.npz")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    half = args.bodies // 2
    ca = torch.tensor([500.0 - args.separation / 2, 500.0 - args.impact / 2], device=dev)
    cb = torch.tensor([500.0 + args.separation / 2, 500.0 + args.impact / 2], device=dev)
    pa, va = init_cluster(half, ca, args.sigma, args.eps, gen, dev)
    pb, vb = init_cluster(half, cb, args.sigma, args.eps, gen, dev)
    va[:, 0] += args.approach
    vb[:, 0] -= args.approach
    pos, vel = torch.cat([pa, pb]), torch.cat([va, vb])
    dyn = CentralForceDynamics(dt=args.dt).to(dev)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    truth, tp, tv = rollout(make_accel("exact", analytic_force(args.eps), 0, 0), pos.clone(), vel.clone(), args.steps, args.dt, args.record)
    print("truth done", flush=True)
    model, mp, mv = rollout(make_accel("far", model_force(dyn), 4.0, args.grid), pos.clone(), vel.clone(), args.steps, args.dt, args.record)
    print("model done", flush=True)
    e0 = energy(pos, vel, args.eps)
    print(f"E0 {e0:.5g} truth end {energy(tp, tv, args.eps):.5g} model end {energy(mp, mv, args.eps):.5g}", flush=True)
    np.savez_compressed(args.out, truth=truth, model=model, dt=args.dt, record=args.record, bodies=args.bodies, half=half)


if __name__ == "__main__":
    main()
