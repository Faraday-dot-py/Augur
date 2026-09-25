"""Accuracy of near (cutoff) + particle-mesh far field vs the exact all-pairs
analytic force on a uniform scaled-init cloud. Prints relative error of the
acceleration for near-only and near+far at each grid size."""
import argparse

import torch

from scripts.gravity_1b import analytic_force, far_accel, init_state, tiled_accel


@torch.no_grad()
def exact_accel(force_fn, pos, chunk=2000):
    acc = torch.zeros_like(pos)
    for i in range(0, pos.shape[0], chunk):
        rel = pos[None, :, :] - pos[i:i + chunk, None, :]
        d = torch.sqrt((rel ** 2).sum(-1, keepdim=True) + 1e-12)
        f = force_fn(d) / d
        f = torch.where(d > 1e-5, f, torch.zeros_like(f))
        acc[i:i + chunk] = (f * rel).sum(1)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=20000)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    args = ap.parse_args()
    dev = torch.device("cuda")
    pos, vel, spread = init_state(args.bodies, dev, args.seed)
    force = analytic_force(args.eps)
    ex = exact_accel(force, pos)
    norm = ex.norm(dim=1)
    print(f"N {args.bodies} spread {spread:.1f} |a| mean {norm.mean():.4g}")
    near = tiled_accel(force, pos, args.radius, 4)
    e = (near - ex).norm(dim=1)
    print(f"near only: mean rel err {float(e.mean() / norm.mean()):.4f}")
    for g in args.grids:
        a = near + far_accel(force, pos, args.radius, g)
        e = (a - ex).norm(dim=1)
        print(f"grid {g}: mean rel err {float(e.mean() / norm.mean()):.4f}  p95 {float(e.quantile(0.95) / norm.mean()):.4f}")


if __name__ == "__main__":
    main()
