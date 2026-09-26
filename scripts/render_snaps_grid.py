"""Frame grid for a 3D orbit_3d.py snapshot npz, for unbiased review: row 1 x-y projection, row 2 x-z, one column per selected frame.
Log density of a fixed square window (--half units) around the median; per-panel auto-scaled.

Usage: PYTHONPATH=. python3 scripts/render_snaps_grid.py results/orbit_X_snaps.npz out.png [--half 900] [--cols 8]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out")
ap.add_argument("--half", type=float, default=900)
ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--dt", type=float, default=0.01)
args = ap.parse_args()
d = np.load(args.npz)
pos, steps = d["pos"], d["steps"]
idx = np.linspace(0, len(pos) - 1, args.cols).round().astype(int)
fig, axs = plt.subplots(2, args.cols, figsize=(2.4 * args.cols, 5.2), facecolor="black")
for j, i in enumerate(idx):
    c = np.median(pos[i], axis=0)
    for r, (a, b, name) in enumerate(((0, 1, "x-y"), (0, 2, "x-z"))):
        e = [np.linspace(c[k] - args.half, c[k] + args.half, 257) for k in (a, b)]
        h = np.histogram2d(pos[i][:, a], pos[i][:, b], bins=e)[0]
        ax = axs[r, j]
        ax.imshow(np.log1p(h).T, origin="lower", cmap="inferno")
        ax.axis("off")
        if r == 0:
            ax.set_title(f"step {steps[i]}", color="white", fontsize=9)
        if j == 0:
            ax.text(0.02, 0.02, name, color="white", transform=ax.transAxes, fontsize=9)
plt.tight_layout()
fig.savefig(args.out, dpi=110, facecolor="black")
