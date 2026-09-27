"""Train LJForceDynamics on small periodic LJ boxes (N=168) by unrolled free rollout of k frames.
Densities in the held-out band are never used for training."""
import argparse
import json
import math
import time

import numpy as np
import torch

from model.lj_force import LJForceDynamics
from scripts import lj_sim

HELD_OUT_RHO = (0.62, 0.82)
TMAX = 2.5
HOT_FRAC = 0.0
HOT_RHO = (0.9, 1.05)
HOT_TMAX = 6.0
FRAME_STEPS = 20


def sample_state(rng):
    """Hot/dense scenes (a HOT_FRAC fraction) push the Maxwell tail into the repulsive core (d < 1.2,
    rarely visited by equilibrated liquid/gas states) so the learned f(d) gets supervision there too --
    see the LJ scaling entry in docs/debugging/experiment-log.md."""
    if rng.random() < HOT_FRAC:
        return float(rng.uniform(*HOT_RHO)), float(rng.uniform(2.0, HOT_TMAX))
    while True:
        rho = float(np.exp(rng.uniform(math.log(0.05), math.log(1.05))))
        if not HELD_OUT_RHO[0] <= rho <= HELD_OUT_RHO[1]:
            break
    temp = float(rng.uniform(0.2, TMAX))
    return rho, temp


def make_data(scenes, frames, warmup, dt, seed, device):
    rng = np.random.default_rng(seed)
    P, V, B, meta = [], [], [], []
    for _ in range(scenes):
        rho, temp = sample_state(rng)
        pos, box = lj_sim.lattice(12, 14, rho, 0.02, rng)
        vel = lj_sim.maxwell(len(pos), temp, rng)
        fp, fv, _ = lj_sim.run(pos, vel, box, warmup + frames * FRAME_STEPS, FRAME_STEPS, dt, device)
        k0 = warmup // FRAME_STEPS
        P.append(fp[k0:])
        V.append(fv[k0:])
        B.append(box)
        meta.append((rho, temp))
    return np.stack(P), np.stack(V), np.stack(B), meta


def wrap_diff(a, b, box):
    d = a - b
    return d - box[..., None, :] * torch.round(d / box[..., None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=240)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--vel-weight", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--tmax", type=float, default=2.5)
    ap.add_argument("--hot-frac", type=float, default=0.0)
    ap.add_argument("--hot-tmax", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default="checkpoints/lj_force.pt")
    ap.add_argument("--out", default="results/lj_train.json")
    args = ap.parse_args()

    global TMAX, HOT_FRAC, HOT_TMAX
    TMAX = args.tmax
    HOT_FRAC = args.hot_frac
    HOT_TMAX = args.hot_tmax
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    t0 = time.time()
    P, V, B, meta = make_data(args.scenes, args.frames, args.warmup, args.dt, args.seed, dev)
    print(f"data {P.shape} in {time.time() - t0:.0f}s", flush=True)
    P = torch.tensor(P, dtype=torch.float64, device=dev)
    V = torch.tensor(V, dtype=torch.float64, device=dev)
    B = torch.tensor(B, dtype=torch.float64, device=dev)
    S, F = P.shape[0], P.shape[1]

    model = LJForceDynamics(dt=args.dt, substeps=FRAME_STEPS, width=args.width).double().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    log = []
    for it in range(args.iters):
        k = 1 + int((args.k_max - 1) * min(1.0, it / (0.6 * args.iters)))
        lr = 1e-4 + 0.5 * (args.lr - 1e-4) * (1 + math.cos(math.pi * it / args.iters))
        for g in opt.param_groups:
            g["lr"] = lr
        si = torch.tensor(rng.integers(0, S, args.batch), device=dev)
        fi = torch.tensor(rng.integers(0, F - k, args.batch), device=dev)
        pos, vel, box = P[si, fi], V[si, fi], B[si]
        ps, vs = model.rollout(pos, vel, box, k)
        loss = 0.0
        for j in range(k):
            tp, tv = P[si, fi + j + 1], V[si, fi + j + 1]
            loss = loss + (wrap_diff(ps[j], tp, box) ** 2).sum(-1).mean() * 100 + args.vel_weight * ((vs[j] - tv) ** 2).sum(-1).mean()
        loss = loss / k
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if it % 25 == 0 or it == args.iters - 1:
            print(f"it {it} k {k} lr {lr:.1e} loss {loss.item():.5f} t {time.time() - t0:.0f}s", flush=True)
            log.append((it, k, loss.item()))
        if it % 250 == 249 or it == args.iters - 1:
            torch.save({"state": model.state_dict(), "args": vars(args), "iter": it}, args.checkpoint)
    json.dump({"log": log, "held_out_rho": HELD_OUT_RHO, "scenes": meta[:5], "args": vars(args)}, open(args.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
