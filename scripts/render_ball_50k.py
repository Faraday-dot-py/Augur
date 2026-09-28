"""Truth (left) vs conservative-contact token model (right) for the 50k-ball
bouncing-ball run; +x is down.
Usage: PYTHONPATH=. python3 scripts/render_ball_50k.py results/ball_50k_rollout.npz videos/ball_50k.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
n = float(d["n"])
fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), dpi=100)
arts = []
for ax, key, name in zip(axes, ("truth_pos", "model_pos"), ("ground truth", "token model (cons_pure_100)")):
    arr = d[key]
    ax.set_xlim(0, n - 1); ax.set_ylim(n - 1, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name)
    arts.append((ax.scatter(arr[0, :, 1], arr[0, :, 0], s=0.3, c="k", alpha=0.5), arr))
title = fig.suptitle("")


def update(i):
    for sc, arr in arts:
        sc.set_offsets(arr[i][:, ::-1])
    title.set_text(f"50000 balls, step {i}")


ani = animation.FuncAnimation(fig, update, frames=d["truth_pos"].shape[0])
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=15, bitrate=4000))
