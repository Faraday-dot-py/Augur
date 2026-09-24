"""Ground truth (top row) and model (bottom row) as separate panels, same axes,
for the N-body gravity test. Bodies are drawn as filled dots, one colour per body.

Usage: PYTHONPATH=. python3 scripts/render_gravity_split.py videos/gravity_test_v2_split.mp4 checkpoints/gravity_dynamics_v2.pt 100,300,1000
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs
from scripts.train_gravity_dynamics import unroll

STEPS, TRAIN_H, DT, EPS = 60, 20, 0.1, 0.5
out, ckpt = sys.argv[1], sys.argv[2]
counts = [int(c) for c in sys.argv[3].split(",")]
rng = np.random.default_rng(9000)
scenes = []
for n in counts:
    p, v = gs.init_bodies(n, rng, scale=True)
    scenes.append(gs.rollout(p, v, STEPS, dt=DT, eps=EPS))

dyn = TokenFreeDynamics(n=1000, neighbor_radius=100.0, pair_impulse=True)
dyn.load_state_dict(torch.load(ckpt))
model = []
with torch.no_grad():
    for P, V in scenes:
        ps, _ = unroll(dyn, torch.tensor(P[0], dtype=torch.float32), torch.tensor(V[0], dtype=torch.float32), STEPS, DT)
        model.append(np.concatenate([P[:1], ps.numpy()]))

fig, axes = plt.subplots(2, len(counts), figsize=(6 * len(counts), 12), dpi=90)
arts = []
for j, ((P, V), M) in enumerate(zip(scenes, model)):
    both = np.concatenate([P, M])
    lo, hi = both.min((0, 1)), both.max((0, 1))
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05 + 0.5
    colors = plt.cm.tab10(np.arange(P.shape[1]) % 10)
    size = max(3, 60 // max(1, P.shape[1] // 8))
    for i, (arr, name) in enumerate(((P, "ground truth"), (M, "model"))):
        ax = axes[i, j]
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{P.shape[1]} bodies: {name}")
        arts.append((ax.scatter(arr[0, :, 0], arr[0, :, 1], s=size, c=colors), arr))
title = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.96))


def update(i):
    for sc, arr in arts:
        sc.set_offsets(arr[i])
    title.set_text(f"N-body gravity, step {i}" + ("  (beyond training horizon)" if i > TRAIN_H else ""))


ani = animation.FuncAnimation(fig, update, frames=STEPS + 1)
ani.save(out, writer=animation.FFMpegWriter(fps=8, bitrate=3000))
