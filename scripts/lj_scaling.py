"""O(N) + tiling validation for LJForceDynamics.accel_single (periodic cell list):
1. Correctness: cell-list accel vs dense chunked all-pairs accel on the same small system.
2. Wall-clock scaling: force-eval time vs N at fixed density (should be ~linear, not quadratic).
3. Tiling: replicate the same unit cell to different N at fixed density/temperature and check
   intensive quantities (T, P, psi6, g(r)) agree -- the model has no notion of box size."""
import argparse
import json
import time

import numpy as np
import torch

from model.lj_force import LJForceDynamics
from scripts import lj_eval as ev
from scripts import lj_sim


def replicate(nx, ny, rho, rng, reps):
    """Tile a jittered lattice `reps` x `reps` times (periodic, so tiles join seamlessly)."""
    pos, box = lj_sim.lattice(nx, ny, rho, 0.02, rng)
    tiles = [pos + box * np.array([i, j]) for i in range(reps) for j in range(reps)]
    return np.concatenate(tiles, 0), box * reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/lj_force_v3.pt")
    ap.add_argument("--rho", type=float, default=0.70)
    ap.add_argument("--temp", type=float, default=0.60)
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--nx", type=int, default=12)
    ap.add_argument("--ny", type=int, default=14)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--rec", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/lj_scaling.json")
    ap.add_argument("--snaps-out", default="results/lj_scaling_big_snaps.npz")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=dev)
    model = LJForceDynamics(dt=args.dt, substeps=20, width=ck["args"].get("width", 64)).double().to(dev)
    model.load_state_dict(ck["state"])
    model.requires_grad_(False)
    model.build_table()

    # 1. correctness: cell-list vs dense, same small system, a few random configs
    rng = np.random.default_rng(args.seed)
    pos, box = lj_sim.lattice(args.nx, args.ny, args.rho, 0.02, rng)
    p = torch.tensor(pos, dtype=torch.float64, device=dev)
    b = torch.tensor(box, dtype=torch.float64, device=dev)
    a_dense = model.accel(p[None], b[None])[0]
    a_cells = model.accel_single(p, b)
    correctness = float((a_dense - a_cells).abs().max())
    print("correctness max|a_dense - a_cells| =", correctness, flush=True)

    # 2. wall-clock scaling (cell-list path) at fixed density
    timing = []
    for reps in args.reps:
        pos_r, box_r = replicate(args.nx, args.ny, args.rho, np.random.default_rng(args.seed), reps)
        p = torch.tensor(pos_r, dtype=torch.float64, device=dev)
        b = torch.tensor(box_r, dtype=torch.float64, device=dev)
        for _ in range(3):
            model.accel_single(p, b)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.time()
        reps_n = max(1, 2000000 // len(pos_r))
        for _ in range(reps_n):
            model.accel_single(p, b)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        dt_call = (time.time() - t0) / reps_n
        timing.append({"n": len(pos_r), "ms_per_eval": dt_call * 1000, "ms_per_particle_us": dt_call / len(pos_r) * 1e6})
        print(timing[-1], flush=True)

    # 3. tiling: short MD at each N via the cell-list path, compare intensive stats
    tiling = []
    big_snaps = None
    for reps in args.reps:
        rng = np.random.default_rng(args.seed)
        pos_r, box_r = replicate(args.nx, args.ny, args.rho, rng, reps)
        vel_r = lj_sim.maxwell(len(pos_r), args.temp, rng)
        p, v, b = (torch.tensor(x, dtype=torch.float64, device=dev) for x in (pos_r, vel_r, box_r))
        for _ in range(args.warmup):
            p, v = model.step_frame_single(p, v, b)
        frames_p, stats = [], []
        for step in range(args.frames):
            p, v = model.step_frame_single(p, v, b)
            if step % 5 == 0:
                a, pe, vir = model.accel_single(p, b, stats=True)
                ke = 0.5 * (v ** 2).sum()
                n = len(pos_r)
                stats.append([float(ke) / n, float(pe) / n, (float(ke) + 0.5 * float(vir)) / float(b[0] * b[1])])
                frames_p.append(p.cpu().numpy().astype(np.float32))
        frames_p = np.stack(frames_p)
        st = np.array(stats)
        g, psi = ev.gr_psi6_scalable(frames_p[-10:], box_r, dev)
        tiling.append({"n": len(pos_r), "reps": reps, "T": float(st[:, 0].mean()), "P": float(st[:, 2].mean()),
                       "psi6": psi, "g": g})
        print(tiling[-1]["n"], "T", tiling[-1]["T"], "P", tiling[-1]["P"], "psi6", psi, flush=True)
        if reps == max(args.reps):
            big_snaps = (frames_p, box_r)

    json.dump({"correctness_max_abs": correctness, "timing": timing, "tiling": tiling,
               "rho": args.rho, "temp": args.temp}, open(args.out, "w"))
    if big_snaps is not None:
        fp, box_r = big_snaps
        np.savez_compressed(args.snaps_out, pos=fp, box=box_r, dt=args.dt * 5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
