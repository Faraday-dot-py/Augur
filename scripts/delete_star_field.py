"""Delete-the-star scene on a trained scatter-field checkpoint, with the force field measured on a grid of test masses.

Each grid point is its own scene: star mass --mstar at the origin plus 4 test masses 0.01 held at that point rotated by 90 degrees (as in the delete-star probe).
The field at the point is the mean of the 4 accelerations rotated back, which cancels the model's uniform self-force offset on the star (star deleted at step --warm).
Saves acc[T, ny, nx, 2], plus the analytic truth.

Usage: PYTHONPATH=. python scripts/delete_star_field.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --tag delete_star_field_E
"""
import argparse

import numpy as np
import torch

from scripts import train_scatter_field as tsf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", default="delete_star_field")
ap.add_argument("--mstar", type=float, default=20.0)
ap.add_argument("--warm", type=int, default=15)
ap.add_argument("--after", type=int, default=25)
ap.add_argument("--half", type=float, default=12.0)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--chunk", type=int, default=144)
args = ap.parse_args()
dev = torch.device("cuda")
model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
model.eval()

xs = (np.arange(args.n) + 0.5) / args.n * 2 * args.half - args.half
X, Y = np.meshgrid(xs, xs)
pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=torch.float32, device=dev)
M = len(pts)
T = args.warm + args.after
acc = np.zeros((T, M, 2), dtype=np.float32)
for s in range(0, M, args.chunk):
    p = pts[s:s + args.chunk]
    B = len(p)
    pos0 = torch.zeros(B, 5, 2, device=dev)
    for k in range(4):
        c, sn = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        pos0[:, 1 + k] = torch.stack([c * p[:, 0] - sn * p[:, 1], sn * p[:, 0] + c * p[:, 1]], -1)
    pos, vel = pos0.clone(), torch.zeros_like(pos0)
    mass = torch.full((B, 5), 0.01, device=dev)
    mass[:, 0] = args.mstar
    mask = torch.ones(B, 5, device=dev)
    field = model.init_field(B, dev)
    with torch.no_grad():
        for t in range(T):
            if t == args.warm:
                mass[:, 0] = 0.0
                mask[:, 0] = 0.0
            pos, vel, field, dv = model.step(pos, vel, mass, mask, field)[:4]
            a = dv[:, 1:] / model.dt
            tot = 0
            for k in range(4):
                c, sn = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
                tot = tot + torch.stack([c * a[:, k, 0] + sn * a[:, k, 1], -sn * a[:, k, 0] + c * a[:, k, 1]], -1)
            acc[t, s:s + B] = (tot / 4).cpu().numpy()
            pos, vel = pos0.clone(), torch.zeros_like(vel)
acc = acc.reshape(T, args.n, args.n, 2)
r2 = X ** 2 + Y ** 2 + 0.25
truth = np.stack([-args.mstar * X / r2 ** 1.5, -args.mstar * Y / r2 ** 1.5], -1)
np.savez(f"results/{args.tag}.npz", xs=xs, acc=acc, truth=truth, warm=args.warm, mstar=args.mstar)
rel = np.linalg.norm(acc[args.warm - 1] - truth, axis=-1) / np.linalg.norm(truth, axis=-1)
print("pre-deletion relative error median/90th", float(np.median(rel)), float(np.percentile(rel, 90)))
print("post-deletion |a| max per step", [round(float(np.linalg.norm(acc[t], axis=-1).max()), 4) for t in range(args.warm, T)])
