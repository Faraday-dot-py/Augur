"""Truth (left) vs token model (right) for the 1000-ball bouncing-ball run;
+x is down. Usage: PYTHONPATH=. python3 scripts/render_ball_1k.py results/ball_1k_rollout_guard.npz videos/ball_1k_500.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
n = float(d["n"])
fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), dpi=80)
arts = []
for ax, key, name in zip(axes, ("truth_pos", "model_pos"), ("ground truth", "token model (soup B)")):
    arr = d[key]
    ax.set_xlim(0, n - 1); ax.set_ylim(n - 1, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name)
    arts.append((ax.scatter(arr[0, :, 1], arr[0, :, 0], s=6, c="k"), arr))
title = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.95))


def update(i):
    for sc, arr in arts:
        sc.set_offsets(arr[i][:, ::-1])
    title.set_text(f"1000 balls, step {i}")


ani = animation.FuncAnimation(fig, update, frames=range(0, d["truth_pos"].shape[0], 2))
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=25, bitrate=3000))
