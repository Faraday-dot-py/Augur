"""Mean kinetic energy per ball vs tick for the free-rollout model on a
uniform-density (0.01) box at several N, with global or per-destination
softmax max. Start is either uniform random velocities or the collision-front
start of scripts/tiled_video.py.

Usage: PYTHONPATH=. python3 scripts/probe_ke_scale.py --counts 100,10000,100000 --ticks 60
"""
import argparse
import math

import torch

from scripts.eval_free_rollout import load_model


def run(model, count, grid, ticks, front, seed):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(count, 2, generator=gen) * (grid - 3) + 1.5
    vel = (torch.rand(count, 2, generator=gen) - 0.5) * 4.6
    if front:
        vel[:, 0] = torch.where(pos[:, 0] < grid / 2, 2.3, -2.3) + vel[:, 0] * 0.2
    hid = torch.zeros(count, model.dynamics.hidden_dim)
    out = []
    with torch.no_grad():
        for t in range(ticks + 1):
            if t % 10 == 0:
                out.append(round(float((vel ** 2).sum(1).mean() / 2), 2))
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            pos = pos.clamp(0.0, grid - 1.0)
            vel = vel.clamp(-30, 30)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--counts", default="100,10000,100000")
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=4738)
    args = ap.parse_args()
    for count in [int(c) for c in args.counts.split(",")]:
        grid = max(100, math.ceil(10 * math.sqrt(count)))
        for local in (False, True):
            for front in (False, True):
                model = load_model(args.checkpoint, "free", grid, 32, 4.0, False, True, True, True, True, True, True)
                model.dynamics.cell_graph = True
                model.dynamics.local_softmax = local
                ke = run(model, count, grid, args.ticks, front, args.seed)
                print(f"N={count} grid={grid} local_softmax={local} front={front} KE/ball@0,10,..: {ke}", flush=True)


if __name__ == "__main__":
    main()
