"""1000-ball x 500-step bouncing run for several model specs against one shared
truth (same setup as scripts/ball_1k_rollout.py, unguarded). Saves one npz with
truth and each model's positions/velocities, and prints per-step KE/ball, error,
floor fraction (x > n-4, gravity is +x), first non-finite step.

Usage: PYTHONPATH=. python3 scripts/ball_1k_fix.py --models soupb=soupb:checkpoints/token_model_soup_b.pt \
    cons=cons:checkpoints/cons_pure.pt --steps 500 --out results/ball_1k_fix.npz
"""
import argparse
import json
import math
import random

import numpy as np
import torch

from model.dataset import make_scenario_uniform
from scripts.ball_1k_rollout import truth
from scripts.energy_fix_common import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--balls", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/ball_1k_fix.npz")
    args = ap.parse_args()
    dev = args.device
    n = max(100, math.ceil(10 * math.sqrt(args.balls)))
    balls = make_scenario_uniform(args.balls, n, 2.3, random.Random(args.seed))
    tp, tv = truth(balls, n, args.steps, 0.15, 9.0, 0.75, 400.0, 8, dev)
    print(f"truth done n={n}", flush=True)
    saved = {"truth_pos": tp, "truth_vel": tv, "n": n}
    summary = {}
    for spec in args.models:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        kw = {}
        if len(parts) > 2:
            kw["adaptive_radius"] = parts[2]
        model = build_model(parts[0], parts[1], n, dev, **kw)
        model.dynamics.cell_graph = True
        pos, vel = torch.from_numpy(tp[0]).clone().to(dev), torch.from_numpy(tv[0]).clone().to(dev)
        hid = torch.zeros(args.balls, model.dynamics.hidden_dim, device=dev)
        mp, mv, first_bad = [pos.cpu().numpy()], [vel.cpu().numpy()], None
        with torch.no_grad():
            for t in range(1, args.steps + 1):
                pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
                if not (bool(torch.isfinite(pos).all()) and float(pos.abs().max()) < 1e4):
                    first_bad = t
                    print(f"{name}: non-finite or > 1e4 at step {t}; stopping", flush=True)
                    break
                mp.append(pos.cpu().numpy())
                mv.append(vel.cpu().numpy())
        mp, mv = np.array(mp), np.array(mv)
        saved[name + "_pos"], saved[name + "_vel"] = mp, mv
        rows = {}
        print(f"== {name}\nstep  err  KE/ball_true  KE/ball_model  floor_frac_true  floor_frac_model  outside  maxspeed_model")
        for t in [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]:
            if t >= len(mp):
                break
            err = float(np.linalg.norm(mp[t] - tp[t], axis=1).mean())
            ket, kem = float((tv[t] ** 2).sum(1).mean() / 2), float((mv[t] ** 2).sum(1).mean() / 2)
            ft, fm = float((tp[t][:, 0] > n - 4).mean()), float((mp[t][:, 0] > n - 4).mean())
            outside = int(((mp[t] < 0) | (mp[t] > n - 1)).any(1).sum())
            ms = float(np.linalg.norm(mv[t], axis=1).max())
            rows[str(t)] = {"err": err, "ke_true": ket, "ke_model": kem, "floor_true": ft, "floor_model": fm, "outside": outside,
                            "maxspeed": ms}
            print(f"{t:4d} {err:8.3f} {ket:10.1f} {kem:10.1f} {ft:8.3f} {fm:8.3f} {outside:5d} {ms:8.1f}", flush=True)
        summary[name] = {"first_nonfinite": first_bad, "rows": rows}
    np.savez(args.out, **saved)
    json.dump(summary, open(args.out.replace(".npz", ".json"), "w"))
    print("done")


if __name__ == "__main__":
    main()
