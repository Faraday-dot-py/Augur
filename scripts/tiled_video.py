"""Runs the tiled sim at N balls and records downsampled frames: a global
density / mean-speed map (`--bins` x `--bins`) and a 1-unit-per-pixel crop
straddling the first strip boundary. Start: left half drifts right, right
half drifts left (a collision front), density 0.01.

Usage:
    PYTHONPATH=. python3 scripts/tiled_video.py --out results/tiled_video_10m.npz
    python3 scripts/render_tiled_video.py results/tiled_video_10m.npz videos/tiled_10m.mp4
"""
import argparse
import math
import time

import numpy as np
import torch

from scripts.eval_free_rollout import load_model
from scripts.tiled_sim import TiledSim


def grid_frames(sim, bins, crop_lo, crop_size, device):
    n = sim.n
    dens = torch.zeros(bins * bins, device=device)
    spd = torch.zeros(bins * bins, device=device)
    crop = torch.zeros(crop_size * crop_size, device=device)
    for s in range(sim.strips):
        c = sim.cores[s][:sim.counts[s]]
        ix = (c[:, 0] / n * bins).long().clamp(0, bins - 1)
        iy = (c[:, 1] / n * bins).long().clamp(0, bins - 1)
        idx = ix * bins + iy
        dens += torch.bincount(idx, minlength=bins * bins).float()
        spd += torch.bincount(idx, weights=c[:, 2:4].norm(dim=1), minlength=bins * bins)
        cx = (c[:, 0] - crop_lo[0]).floor().long()
        cy = (c[:, 1] - crop_lo[1]).floor().long()
        ok = (cx >= 0) & (cx < crop_size) & (cy >= 0) & (cy < crop_size)
        crop += torch.bincount(cx[ok] * crop_size + cy[ok], minlength=crop_size * crop_size).float()
    return (dens.view(bins, bins).cpu().numpy().astype(np.uint16),
            (spd / dens.clamp(min=1)).view(bins, bins).cpu().numpy().astype(np.float16),
            crop.view(crop_size, crop_size).cpu().numpy().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--balls", type=int, default=10_000_000)
    ap.add_argument("--tile-balls", type=int, default=4_000_000)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--bins", type=int, default=512)
    ap.add_argument("--crop", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    grid = math.ceil(10 * math.sqrt(args.balls))
    model = load_model(args.checkpoint, "free", grid, 32, 4.0, False, True, True, True, True, True, True).to(device)
    sim = TiledSim(model, math.ceil(args.balls / args.tile_balls), device)
    sim.init_random(args.balls, args.seed)
    for s in range(sim.strips):
        c = sim.cores[s][:sim.counts[s]]
        c[:, 2] = torch.where(c[:, 0] < grid / 2, 2.3, -2.3) + c[:, 2] * 0.2
    crop_lo = (sim.lo(1) - args.crop / 2, grid / 2 - args.crop / 2)

    dens, spd, crop, ke, alive = [], [], [], [], []
    start = time.perf_counter()
    for t in range(args.ticks + 1):
        d, v, c = grid_frames(sim, args.bins, crop_lo, args.crop, device)
        dens.append(d), spd.append(v), crop.append(c)
        alive.append(sim.alive())
        ke.append(sum(float((sim.cores[s][:sim.counts[s], 2:4] ** 2).sum()) for s in range(sim.strips)) / 2)
        if t % 10 == 0:
            print(f"tick {t} alive {alive[-1]} ke {ke[-1]:.4g} max/bin {d.max()} elapsed {time.perf_counter() - start:.0f} s", flush=True)
        if t < args.ticks:
            sim.step()
    np.savez_compressed(args.out, density=np.stack(dens), speed=np.stack(spd), crop=np.stack(crop),
                        ke=np.array(ke), alive=np.array(alive), grid=grid, strips=sim.strips,
                        boundary=sim.lo(1), crop_lo=np.array(crop_lo), balls=args.balls)


if __name__ == "__main__":
    main()
