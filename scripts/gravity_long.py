"""Long N-body gravity rollout: ground truth and model (free-running from step 0)
side by side. Positions are cached in an npz (recorded every --every steps), then
rendered with a window that follows the bodies' 1-99 percentile extent (from the
ground truth, smoothed) so it fits the particle cloud at every frame.

Usage: PYTHONPATH=. python3 scripts/gravity_long.py --out videos/gravity_long.mp4
"""
import argparse
import os

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs


def accel_t(pos, eps):
    d = pos[None, :, :] - pos[:, None, :]
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
    inv.fill_diagonal_(0.0)
    return (d * inv[..., None]).sum(1)


def energy_t(pos, vel, eps):
    d = pos[None, :, :] - pos[:, None, :]
    r = ((d ** 2).sum(-1) + eps ** 2).sqrt()
    iu = torch.triu_indices(len(pos), len(pos), 1, device=pos.device)
    return float(0.5 * (vel ** 2).sum() - (1.0 / r[iu[0], iu[1]]).sum())


def simulate(args):
    rng = np.random.default_rng(args.seed)
    p0, v0 = gs.init_bodies(args.bodies, rng, scale=True)
    h = args.dt / 4
    dev = torch.device(args.device)
    pos, vel = torch.tensor(p0, device=dev), torch.tensor(v0, device=dev)
    a = accel_t(pos, args.eps)
    if args.model == "central":
        dyn = CentralForceDynamics(dt=args.dt, neighbor_radius=100.0).to(dev)
    else:
        dyn = TokenFreeDynamics(n=1000, neighbor_radius=100.0, pair_impulse=True).to(dev)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    mp, mv = torch.tensor(p0, dtype=torch.float32, device=dev), torch.tensor(v0, dtype=torch.float32, device=dev)
    hidden = torch.zeros(args.bodies, dyn.hidden_dim, device=dev)
    truth, model, e_t, e_m = [p0.copy()], [p0.copy()], [], []
    for step in range(1, args.steps + 1):
        for _ in range(4):
            vel += 0.5 * h * a
            pos += h * vel
            a = accel_t(pos, args.eps)
            vel += 0.5 * h * a
        with torch.no_grad():
            dp, dv, hidden = dyn(mp, mv, hidden)
            mp = mp + mv * args.dt + dp
            mv = mv + dv
        if step % args.every == 0:
            truth.append(pos.cpu().numpy().copy())
            model.append(mp.cpu().numpy().astype(np.float64))
            e_t.append(energy_t(pos, vel, args.eps))
            e_m.append(energy_t(mp.double(), mv.double(), args.eps))
            print(f"step {step} truth E {e_t[-1]:.1f} model E {e_m[-1]:.1f} model finite {bool(torch.isfinite(mp).all())}", flush=True)
    np.savez_compressed(args.cache, truth=np.stack(truth), model=np.stack(model), energy_t=e_t, energy_m=e_m, every=args.every)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    d = np.load(args.cache)
    truth, model, every = d["truth"], d["model"], int(d["every"])
    frames = truth.shape[0]
    boxes = []
    for f in range(frames):
        lo, hi = np.percentile(truth[f], 1, axis=0), np.percentile(truth[f], 99, axis=0)
        c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.15 + 1.0
        boxes.append(np.concatenate([c, [half]]))
    boxes = np.array(boxes)
    k = 15
    smooth = np.stack([np.convolve(np.pad(boxes[:, i], k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for i in range(3)], axis=1)
    colors = plt.cm.tab10(np.arange(truth.shape[1]) % 10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), dpi=90)
    scs = []
    for ax, arr, name in zip(axes, (truth, model), ("ground truth", "model")):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name)
        scs.append(ax.scatter(arr[0, :, 0], arr[0, :, 1], s=6, c=colors))
    title = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(i):
        cx, cy, half = smooth[i]
        for ax, sc, arr in zip(axes, scs, (truth, model)):
            sc.set_offsets(arr[i])
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        gap = np.linalg.norm(model[i] - truth[i], axis=1).mean()
        title.set_text(f"{truth.shape[1]} bodies, step {i * every}, window {2 * half:.0f} wide, mean pos gap {gap:.1f}")

    ani = animation.FuncAnimation(fig, update, frames=frames)
    ani.save(args.out, writer=animation.FFMpegWriter(fps=30, bitrate=3000))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_dynamics_v2.pt")
    ap.add_argument("--model", choices=["token", "central"], default="token")
    ap.add_argument("--bodies", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--cache", default="results/gravity_long.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.path.exists(args.cache):
        simulate(args)
    if not args.no_render:
        render(args)
