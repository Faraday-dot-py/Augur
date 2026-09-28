"""Large-N bouncing-ball rollout (square box, cell-list truth, no N x N tensors):
conservative-contact model vs truth, gravity 9 toward +x. Same physics as
scripts/ball_1k_rollout.py but truth pair forces use build_radius_graph_cells
(O(N)) instead of a dense pairwise matrix, so this scales to 50k+ balls.

Usage: PYTHONPATH=. python3 scripts/ball_50k_rollout.py --balls 50000 --steps 100 \
    --ckpt checkpoints/cons_pure_100.pt --out results/ball_50k_rollout.npz
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch

from model.dataset import make_scenario_uniform
from model.token_graph import build_radius_graph_cells
from scripts.energy_fix_common import build_model


def pen(x, k):
    return torch.where(x > 0.0, k * x * x, torch.zeros_like(x))


def pair_edges(p, radius):
    e = build_radius_graph_cells(p, 2 * radius)
    src, dst = e[0], e[1]
    d = p[dst] - p[src]
    dist = torch.sqrt((d ** 2).sum(-1) + 1e-18)
    return src, dst, d, dist


def forces(p, n, gravity, radius, k):
    lo, hi = 0.0, n - 1.0
    x, y = p[:, 0], p[:, 1]
    left_x = (x - lo) < radius
    right_x = ~left_x & ((hi - x) < radius)
    left_y = (y - lo) < radius
    right_y = ~left_y & ((hi - y) < radius)
    fx = gravity + torch.where(left_x, pen(radius - (x - lo), k), 0.0) - torch.where(right_x, pen(radius - (hi - x), k), 0.0)
    fy = torch.where(left_y, pen(radius - (y - lo), k), 0.0) - torch.where(right_y, pen(radius - (hi - y), k), 0.0)
    f = torch.stack([fx, fy], dim=1)
    src, dst, d, dist = pair_edges(p, radius)
    fm = pen(2 * radius - dist, k) / dist.clamp(min=1e-9)
    return f.index_add(0, src, -fm.unsqueeze(-1) * d)


def run_truth(p, v, n, steps, dt, gravity, radius, k, substeps, dev):
    p, v = p.double().to(dev), v.double().to(dev)
    sub = dt / substeps
    ps, vs = [], []
    for _ in range(steps + 1):
        ps.append(p.float().cpu().numpy())
        vs.append(v.float().cpu().numpy())
        for _ in range(substeps):
            v = v + forces(p, n, gravity, radius, k) * sub
            p = p + v * sub
    return np.array(ps), np.array(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balls", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--vmax", type=float, default=2.3)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--ckpt", default="checkpoints/cons_pure_100.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/ball_50k_rollout.npz")
    args = ap.parse_args()
    dev, N = args.device, args.balls
    dt, radius, k = 0.15, 0.75, 400.0
    n = max(100, math.ceil(10 * math.sqrt(N)))

    balls = make_scenario_uniform(N, n, args.vmax, random.Random(args.seed))
    p0 = torch.tensor([[b["x"], b["y"]] for b in balls])
    v0 = torch.tensor([[b["vx"], b["vy"]] for b in balls])
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tp, tv = run_truth(p0, v0, n, args.steps, dt, 9.0, radius, k, 8, dev)
    t_truth = time.time() - t0
    print(f"truth done n={n} {t_truth:.1f}s", flush=True)

    model = build_model("cons", args.ckpt, n, dev)
    model.dynamics.cell_graph = True
    print(f"model learned gravity = {(model.dynamics.gravity * 10).tolist()}", flush=True)
    pos, vel = torch.from_numpy(tp[0]).to(dev), torch.from_numpy(tv[0]).to(dev)
    hid = torch.zeros(N, model.dynamics.hidden_dim, device=dev)
    mp, mv = [pos.cpu().numpy()], [vel.cpu().numpy()]
    first_bad = None
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for t in range(1, args.steps + 1):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            if not (bool(torch.isfinite(pos).all()) and bool(torch.isfinite(vel).all())):
                first_bad = t
                print(f"model non-finite at step {t}; stopping", flush=True)
                break
            mp.append(pos.cpu().numpy())
            mv.append(vel.cpu().numpy())
            if t % 10 == 0:
                print(f"model step {t} {time.time() - t0:.0f}s", flush=True)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_model = time.time() - t0
    mp, mv = np.array(mp), np.array(mv)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
    np.savez(args.out, truth_pos=tp, truth_vel=tv, model_pos=mp, model_vel=mv, n=n)

    rows = {}
    print("step  err  KE_t  KE_m  floor_t  floor_m  outside_m  maxspeed_m")
    for t in [1, 5, 10, 20, 50, 100, 150, 200]:
        if t >= len(mp):
            break
        err = float(np.linalg.norm(mp[t] - tp[t], axis=1).mean())
        ket, kem = float((tv[t] ** 2).sum(1).mean() / 2), float((mv[t] ** 2).sum(1).mean() / 2)
        ft, fm = float((tp[t][:, 0] > n - 4).mean()), float((mp[t][:, 0] > n - 4).mean())
        out = int(((mp[t] < 0) | (mp[t] > n - 1)).any(1).sum())
        ms = float(np.linalg.norm(mv[t], axis=1).max())
        rows[str(t)] = {"err": err, "ke_t": ket, "ke_m": kem, "floor_t": ft, "floor_m": fm, "outside_m": out, "maxspeed_m": ms}
        print(f"{t:4d} {err:8.3f} {ket:8.2f} {kem:8.2f} {ft:6.3f} {fm:6.3f} {out:5d} {ms:8.1f}", flush=True)
    summ = {"n": n, "balls": N, "first_nonfinite": first_bad, "truth_seconds": t_truth, "model_seconds": t_model,
            "peak_gpu_gib": peak, "rows": rows}
    json.dump(summ, open(args.out.replace(".npz", ".json"), "w"))
    print(f"first_nonfinite={first_bad} truth_s={t_truth:.1f} model_s={t_model:.1f} peak_gpu_gib={peak:.2f}")


if __name__ == "__main__":
    main()
