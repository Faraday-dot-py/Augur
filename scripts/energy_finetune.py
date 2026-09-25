"""State-space fine-tune of soup B's dynamics on GPU-generated contact-rich,
high-speed scenes, to separate data/loss coverage from architecture as the cause
of wall/pair energy errors (docs/debugging/energy-conservation-investigation.md).

Modes: wide (box 20-200, init speed up to 40, 30% replay of the original 20x20
distribution) or ctrl (original distribution only). --unroll K sets the grad
horizon (teacher-forced carried-hidden warm-up, then K free steps).

Usage: PYTHONPATH=. python3 scripts/energy_finetune.py --mode wide --unroll 1 --out checkpoints/energy_ft_wide_k1.pt
"""
import argparse
import random
import time

import numpy as np
import torch

from model.token_losses import token_state_loss
from scripts.energy_probe2 import truth_states
from scripts.eval_free_rollout import load_model

STEPS = 24


def sample_scene(mode, rng):
    if mode == "wall":
        n = rng.choice([20, 40, 80, 160])
        speed = rng.uniform(1.0, 40.0)
        ang = rng.uniform(0, 6.2832)
        return n, [[rng.uniform(2, n - 3.0), rng.uniform(2, n - 3.0)]], [[speed * float(np.cos(ang)), speed * float(np.sin(ang))]]
    if mode == "pair":
        s = rng.uniform(1.0, 12.0)
        off = rng.uniform(-1.4, 1.4)
        ang = rng.uniform(0, 6.2832)
        c, sn = float(np.cos(ang)), float(np.sin(ang))
        d = 6.0
        a = [50.0 - d * c - off * sn, 50.0 - d * sn + off * c]
        b = [50.0 + d * c + off * sn, 50.0 + d * sn - off * c]
        return 100, [a, b], [[s * c, s * sn], [-s * c, -s * sn]]
    if mode == "ctrl" or rng.random() < 0.3:
        n, nb, vmax = 20, rng.randint(2, 6), 2.3
    else:
        n = rng.choice([20, 30, 50, 80, 120, 200])
        nb = max(2, int(n * n * rng.uniform(0.005, 0.02)))
        nb = min(nb, 120)
        vmax = rng.uniform(2.3, 40.0)
    p0 = [[rng.uniform(0, n - 1.0), rng.uniform(0, n - 1.0)] for _ in range(nb)]
    v0 = [[rng.uniform(-vmax, vmax), rng.uniform(-vmax, vmax)] for _ in range(nb)]
    return n, p0, v0


def make_scenes(mode, count, seed, device):
    rng = random.Random(seed)
    scenes = []
    for _ in range(count):
        n, p0, v0 = sample_scene(mode, rng)
        ps, vs = truth_states(p0, v0, n, STEPS, device)
        scenes.append((n, torch.from_numpy(ps).to(device), torch.from_numpy(vs).to(device)))
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--mode", default="wide")
    ap.add_argument("--unroll", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--scenes", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)

    t0 = time.time()
    scenes = make_scenes(args.mode, args.scenes, args.seed, dev)
    print(f"generated {len(scenes)} scenes in {time.time() - t0:.0f}s", flush=True)

    model = load_model(args.checkpoint, "free", 20, 32, 4.0, False, True, True, True, True, True, True).to(dev)
    model.train()
    dyn = model.dynamics
    opt = torch.optim.Adam(dyn.parameters(), lr=args.lr)
    rng = random.Random(args.seed + 1)
    K = args.unroll
    run = 0.0
    for it in range(1, args.iters + 1):
        opt.zero_grad()
        tot = 0.0
        for _ in range(args.batch):
            n, ps, vs = scenes[rng.randrange(len(scenes))]
            dyn.n = n
            dyn.cell_graph = ps.shape[1] > 50
            nb = ps.shape[1]
            hid = torch.zeros(nb, dyn.hidden_dim, device=dev)
            start = rng.randint(0, STEPS - K)
            with torch.no_grad():
                for t in range(start):
                    _, _, hid, _ = model.step_free(ps[t], vs[t], hid, render=False)
            pos, vel = ps[start], vs[start]
            loss = 0.0
            for k in range(K):
                pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
                tgt = {"x": ps[start + k + 1][:, 0], "y": ps[start + k + 1][:, 1],
                       "vx": vs[start + k + 1][:, 0], "vy": vs[start + k + 1][:, 1]}
                loss = loss + token_state_loss(pos, vel, tgt, torch.arange(nb, device=dev), vel_weight=0.1, speed_weight=0.3)
            loss = loss / K / args.batch
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        run = 0.98 * run + 0.02 * tot if it > 1 else tot
        if it % 100 == 0:
            print(f"it {it} loss {tot:.4f} ema {run:.4f} t={time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
