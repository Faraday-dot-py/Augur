"""Export a few eval scenes' truth + token + central rollouts (positions, velocities)
to an npz for local video rendering (see render_gravity_relativistic_video.py).
Must run on GPU -- reuses the same relativistic data-gen and unroll as
train_gravity_relativistic.py."""
import argparse

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs
from scripts.train_gravity_dynamics import unroll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--min-bodies", type=int, default=3)
    ap.add_argument("--max-bodies", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--neighbor-radius", type=float, default=100.0)
    ap.add_argument("--token-ckpt", default="checkpoints/gravity_relativistic_token.pt")
    ap.add_argument("--central-ckpt", default="checkpoints/gravity_relativistic_central.pt")
    ap.add_argument("--out", default="results/gravity_relativistic_rollout.npz")
    args = ap.parse_args()

    dev = torch.device("cuda")
    kw = dict(dt=args.dt, eps=args.eps, relativistic=True, c=args.c)
    data = gs.make_dataset(args.n_scenes, (args.min_bodies, args.max_bodies), args.steps, args.seed, device=dev, **kw)

    token = TokenFreeDynamics(n=1000, neighbor_radius=args.neighbor_radius, pair_impulse=True)
    token.load_state_dict(torch.load(args.token_ckpt, map_location=dev))
    token.to(dev).eval()
    central = CentralForceDynamics(dt=args.dt, neighbor_radius=args.neighbor_radius)
    central.load_state_dict(torch.load(args.central_ckpt, map_location=dev))
    central.to(dev).eval()

    scenes = []
    with torch.no_grad():
        for P, V in data:
            p0 = torch.tensor(P[0], dtype=torch.float32, device=dev)
            v0 = torch.tensor(V[0], dtype=torch.float32, device=dev)
            tp, tv = unroll(token, p0, v0, args.steps, args.dt)
            cp, cv = unroll(central, p0, v0, args.steps, args.dt)
            scenes.append({
                "truth_pos": P, "truth_vel": V,
                "token_pos": np.concatenate([P[:1], tp.cpu().numpy()], 0),
                "token_vel": np.concatenate([V[:1], tv.cpu().numpy()], 0),
                "central_pos": np.concatenate([P[:1], cp.cpu().numpy()], 0),
                "central_vel": np.concatenate([V[:1], cv.cpu().numpy()], 0),
            })
    np.savez(args.out, c=args.c, n_scenes=args.n_scenes,
             **{f"{k}_{i}": v for i, s in enumerate(scenes) for k, v in s.items()})
    print(f"saved {len(scenes)} scenes to {args.out}", flush=True)


if __name__ == "__main__":
    main()
