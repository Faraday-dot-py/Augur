"""Truth vs several models for the 1000-ball run (npz from scripts/ball_1k_fix.py);
+x is down. Models that stopped early hold their last frame.
Usage: PYTHONPATH=. python3 scripts/render_ball_1k_compare.py results/ball_1k_fix.npz videos/ball_1k_energy_fix.mp4 [grid.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
n = float(d["n"])
names = ["truth"] + sorted(k[:-4] for k in d.files if k.endswith("_pos") and k != "truth_pos")
T = d["truth_pos"].shape[0]


def frame(name, i):
    arr = d[name + "_pos"]
    return arr[min(i, arr.shape[0] - 1)]


if len(sys.argv) > 3:
    steps = [0, 5, 10, 20, 50, 100, 200, 499]
    fig, axes = plt.subplots(len(names), len(steps), figsize=(3 * len(steps), 3.2 * len(names)), dpi=70)
    for r, name in enumerate(names):
        for c, t in enumerate(steps):
            ax = axes[r, c]
            p = frame(name, t)
            ax.scatter(p[:, 1], p[:, 0], s=3, c="k")
            ax.set_xlim(0, n - 1); ax.set_ylim(n - 1, 0); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{name} step {t}", fontsize=9)
    fig.tight_layout()
    fig.savefig(sys.argv[3])

fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6.4), dpi=60)
arts = []
for ax, name in zip(axes, names):
    ax.set_xlim(0, n - 1); ax.set_ylim(n - 1, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name)
    arts.append((ax.scatter(frame(name, 0)[:, 1], frame(name, 0)[:, 0], s=6, c="k"), name))
title = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.95))


def update(i):
    for sc, name in arts:
        sc.set_offsets(frame(name, i)[:, ::-1])
    title.set_text(f"1000 balls, step {i}")


ani = animation.FuncAnimation(fig, update, frames=range(0, T, 2))
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=25, bitrate=3000))
