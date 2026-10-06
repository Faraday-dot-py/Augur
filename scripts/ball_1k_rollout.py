"""1000-ball bouncing-ball sim (walls, contact forces, gravity 9 toward +x, no
body-to-body gravity) vs the token model's free rollout from the same true
initial state. Saves per-step positions/velocities of both.

Usage: PYTHONPATH=. python3 scripts/ball_1k_rollout.py --steps 500
"""
import argparse
import math
import random

import numpy as np
import torch

import augur
from model.dataset import make_scenario_uniform
from scripts.eval_free_rollout import load_model


def pen(x, k):
    return torch.where(x > 0.0, k * x * x, torch.zeros_like(x))


def forces(p, n, gravity, radius, k):
    lo, hi = 0.0, n - 1.0
    x, y = p[:, 0], p[:, 1]
    left_x = (x - lo) < radius
    right_x = ~left_x & ((hi - x) < radius)
    left_y = (y - lo) < radius
    right_y = ~left_y & ((hi - y) < radius)
    fx = gravity + torch.where(left_x, pen(radius - (x - lo), k), 0.0) - torch.where(right_x, pen(radius - (hi - x), k), 0.0)
    fy = torch.where(left_y, pen(radius - (y - lo), k), 0.0) - torch.where(right_y, pen(radius - (hi - y), k), 0.0)
    d = p[None, :, :] - p[:, None, :]
    dist = d.norm(dim=-1)
    f = pen(2 * radius - dist, k)
    f.fill_diagonal_(0.0)
    f = f / dist.clamp(min=1e-9)
    pf = (f[..., None] * d).sum(0)
    return torch.stack([fx, fy], dim=1) + pf


def truth(balls, n, steps, dt, gravity, radius, stiffness, substeps, device):
    p = torch.tensor([[b["x"], b["y"]] for b in balls], dtype=torch.float64, device=device)
    v = torch.tensor([[b["vx"], b["vy"]] for b in balls], dtype=torch.float64, device=device)
    ps, vs = [], []
    sub = dt / substeps
    for _ in range(steps + 1):
        ps.append(p.float().cpu().numpy())
        vs.append(v.float().cpu().numpy())
        for _ in range(substeps):
            v = v + forces(p, n, gravity, radius, stiffness) * sub
            p = p + v * sub
    return np.array(ps), np.array(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--balls", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--guard", action="store_true", help="realtime_sim.py contain_state-style clamp + speed cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/ball_1k_rollout.npz")
    args = ap.parse_args()

    n = max(100, math.ceil(10 * math.sqrt(args.balls)))
    balls = make_scenario_uniform(args.balls, n, 2.3, random.Random(args.seed))
    tp, tv = truth(balls, n, args.steps, 0.15, 9.0, 0.75, 400.0, 8, args.device)
    print(f"truth done n={n}", flush=True)

    model = load_model(args.checkpoint, "free", n, 32, 4.0, False, True, True, True, True, True, True)
    model.dynamics.cell_graph = True
    model.to(args.device)
    pos, vel = torch.from_numpy(tp[0]).clone().to(args.device), torch.from_numpy(tv[0]).clone().to(args.device)
    hid = torch.zeros(args.balls, model.dynamics.hidden_dim, device=args.device)
    mp, mv = [pos.cpu().numpy()], [vel.cpu().numpy()]
    with torch.no_grad():
        for _ in range(args.steps):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            if args.guard:
                pos, vel = torch.nan_to_num(pos), torch.nan_to_num(vel)
                out = ((pos < 0) & (vel < 0)) | ((pos > n - 1.0) & (vel > 0))
                vel = torch.where(out, torch.zeros_like(vel), vel)
                pos = pos.clamp(0.0, n - 1.0)
                sp = vel.norm(dim=1, keepdim=True).clamp(min=1e-6)
                vel = vel * (sp.clamp(max=30.0) / sp)
            elif not (bool(torch.isfinite(pos).all()) and float(pos.abs().max()) < 1e4):
                print(f"model state non-finite or > 1e4 at step {len(mp)}; stopping", flush=True)
                break
            mp.append(pos.cpu().numpy())
            mv.append(vel.cpu().numpy())
    mp, mv = np.array(mp), np.array(mv)
    np.savez(args.out.replace(".npz", "_guard.npz") if args.guard else args.out, truth_pos=tp, truth_vel=tv, model_pos=mp, model_vel=mv, n=n)

    tp, tv = tp[: len(mp)], tv[: len(mp)]
    cv = tp[0][None] + tv[0][None] * 0.15 * np.arange(len(mp))[:, None, None]
    print("step  err_model  err_constvel  KE/ball_true  KE/ball_model  meanX_true  meanX_model  outside_model")
    for t in [1, 3, 5, 10, 20, 50, 100, 200, 300, 400, 500]:
        if t >= len(mp):
            break
        em = np.linalg.norm(mp[t] - tp[t], axis=1).mean()
        ec = np.linalg.norm(cv[t] - tp[t], axis=1).mean()
        ket, kem = (tv[t] ** 2).sum(1).mean() / 2, (mv[t] ** 2).sum(1).mean() / 2
        out = ((mp[t] < 0) | (mp[t] > n - 1)).any(1).sum()
        print(f"{t:4d}  {em:9.3f}  {ec:12.3f}  {ket:12.3f}  {kem:13.3f}  {tp[t][:, 0].mean():10.1f}  {mp[t][:, 0].mean():11.1f}  {out}")


if __name__ == "__main__":
    main()
