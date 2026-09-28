"""Train the conservative-contact TokenFreeDynamics (learned pair force + wall force
+ uniform acceleration, velocity Verlet sub-steps) on GPU-generated truth scenes
(single-ball wall, head-on/offset pairs, wide multi-ball boxes, original 20x20).
Init is from scratch (zero forces). Speeds in training are capped at 40; 60/80 are
evaluation-only extrapolation.

Usage: PYTHONPATH=. python3 scripts/train_conservative.py --residual 0 --out checkpoints/cons_pure.pt
"""
import argparse
import random
import time

import numpy as np
import torch

from model.token_losses import token_state_loss
from model.token_model import TokenModel
from scripts.energy_probe2 import truth_states

STEPS = 24
VMAX = 40.0


def sample_scene(rng):
    u = rng.random()
    if u < 0.25:
        n = rng.choice([20, 40, 80, 160])
        speed, ang = rng.uniform(1.0, VMAX), rng.uniform(0, 6.2832)
        return n, [[rng.uniform(2, n - 3.0), rng.uniform(2, n - 3.0)]], [[speed * float(np.cos(ang)), speed * float(np.sin(ang))]]
    if u < 0.5:
        s = rng.uniform(1.0, VMAX)
        off = rng.uniform(-1.4, 1.4)
        ang = rng.uniform(0, 6.2832)
        c, sn = float(np.cos(ang)), float(np.sin(ang))
        d = 6.0
        a = [50.0 - d * c - off * sn, 50.0 - d * sn + off * c]
        b = [50.0 + d * c + off * sn, 50.0 + d * sn - off * c]
        return 100, [a, b], [[s * c, s * sn], [-s * c, -s * sn]]
    if u < 0.65:
        n, nb, vmax = 20, rng.randint(2, 6), 2.3
    else:
        n = rng.choice([20, 30, 50, 80, 120, 200])
        nb = min(max(2, int(n * n * rng.uniform(0.005, 0.02))), 120)
        vmax = rng.uniform(2.3, VMAX)
    p0 = [[rng.uniform(0, n - 1.0), rng.uniform(0, n - 1.0)] for _ in range(nb)]
    v0 = [[rng.uniform(-vmax, vmax), rng.uniform(-vmax, vmax)] for _ in range(nb)]
    return n, p0, v0


def make_scenes(count, seed, device):
    rng = random.Random(seed)
    scenes = []
    for _ in range(count):
        n, p0, v0 = sample_scene(rng)
        ps, vs = truth_states(p0, v0, n, STEPS, device)
        scenes.append((n, torch.from_numpy(ps).to(device), torch.from_numpy(vs).to(device)))
    return scenes


def scene_loss(model, scene, start, K, residual, device):
    n, ps, vs = scene
    dyn = model.dynamics
    dyn.n = n
    nb = ps.shape[1]
    dyn.cell_graph = nb > 50
    hid = torch.zeros(nb, dyn.hidden_dim, device=device)
    if residual:
        with torch.no_grad():
            for t in range(start):
                _, _, hid, _ = model.step_free(ps[t], vs[t], hid, render=False)
    pos, vel = ps[start], vs[start]
    loss = 0.0
    for k in range(K):
        pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
        tgt = {"x": ps[start + k + 1][:, 0], "y": ps[start + k + 1][:, 1],
               "vx": vs[start + k + 1][:, 0], "vy": vs[start + k + 1][:, 1]}
        loss = loss + token_state_loss(pos, vel, tgt, torch.arange(nb, device=device), vel_weight=0.1, speed_weight=0.3)
    return loss / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", type=int, default=0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--scenes", type=int, default=1200)
    ap.add_argument("--val-scenes", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    t0 = time.time()
    scenes = make_scenes(args.scenes, args.seed, dev)
    val = make_scenes(args.val_scenes, 9000, dev)
    print(f"generated {len(scenes)}+{len(val)} scenes in {time.time() - t0:.0f}s", flush=True)

    model = TokenModel(n=20, radius=0.75, dt=0.15, hidden_dim=32, neighbor_radius=4.0, free_rollout=True,
                       conservative_contact=True, contact_substeps=args.substeps,
                       contact_residual=bool(args.residual)).to(dev)
    dyn = model.dynamics
    fast = [dyn.gravity]
    slow = [p for name, p in dyn.named_parameters() if name != "gravity"]
    opt = torch.optim.Adam([{"params": slow, "lr": args.lr}, {"params": fast, "lr": 3e-2}])
    flags = {"substeps": args.substeps, "residual": bool(args.residual)}
    rng = random.Random(args.seed + 1)
    iters = args.scenes // args.batch
    for epoch in range(args.epochs):
        K = 1 if epoch < 10 else (4 if epoch < 30 else 8)
        scale = 0.1 if epoch >= int(0.8 * args.epochs) else 1.0
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = args.lr * scale, 3e-2 * scale
        order = list(range(len(scenes)))
        rng.shuffle(order)
        run = 0.0
        for it in range(iters):
            opt.zero_grad()
            tot = 0.0
            for j in order[it * args.batch:(it + 1) * args.batch]:
                start = rng.randint(0, STEPS - K)
                loss = scene_loss(model, scenes[j], start, K, args.residual, dev) / args.batch
                loss.backward()
                tot += float(loss)
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
            opt.step()
            run += tot
        msg = f"epoch {epoch} K={K} loss {run / iters:.4f} g={(dyn.gravity * 10).tolist()} t={time.time() - t0:.0f}s"
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            with torch.no_grad():
                vl = np.mean([float(scene_loss(model, s, 0, 8, args.residual, dev)) for s in val])
            msg += f" val8 {vl:.4f}"
        print(msg, flush=True)
        torch.save({"model": model.state_dict(), "flags": flags, "opt": opt.state_dict(), "epoch": epoch}, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
