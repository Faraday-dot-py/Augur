"""Still grid: rows truth/model, columns steps. +x is down.
Usage: PYTHONPATH=. python3 scripts/render_ball_1k_grid.py results/ball_1k_rollout_guard.npz videos/ball_1k_grid.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
n = float(d["n"])
steps = [0, 5, 10, 20, 50, 100, 200, 500]
fig, axes = plt.subplots(2, len(steps), figsize=(3 * len(steps), 6.4), dpi=80)
for r, (key, name) in enumerate((("truth_pos", "truth"), ("model_pos", "model"))):
    for c, t in enumerate(steps):
        ax = axes[r, c]
        p = d[key][t]
        ax.scatter(p[:, 1], p[:, 0], s=3, c="k")
        ax.set_xlim(0, n - 1); ax.set_ylim(n - 1, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name} step {t}", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2])
