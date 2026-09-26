"""Frame grid for a 2D black-hole snapshot npz (orbit_bh.py), for unbiased review: row 1 wide window (--wide half-width), row 2 zoomed window (--zoom), one column per selected frame.
Log density of a square window centred on the median position; per-panel auto-scaled.

Usage: PYTHONPATH=. python3 scripts/render_bh_grid.py results/orbit_bh_X_snaps.npz out.png [--wide 60] [--zoom 4] [--cols 8]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out")
ap.add_argument("--wide", type=float, default=60)
ap.add_argument("--zoom", type=float, default=4)
ap.add_argument("--cols", type=int, default=8)
args = ap.parse_args()
d = np.load(args.npz)
pos, steps = d["pos"], d["steps"]
idx = np.linspace(0, len(pos) - 1, args.cols).round().astype(int)
fig, axs = plt.subplots(2, args.cols, figsize=(2.4 * args.cols, 5.2), facecolor="black")
for j, i in enumerate(idx):
    c = np.median(pos[i], axis=0)
    for r, half in enumerate((args.wide, args.zoom)):
        e = [np.linspace(c[k] - half, c[k] + half, 257) for k in (0, 1)]
        h = np.histogram2d(pos[i][:, 0], pos[i][:, 1], bins=e)[0]
        ax = axs[r, j]
        ax.imshow(np.log1p(h).T, origin="lower", cmap="inferno")
        ax.axis("off")
        if r == 0:
            ax.set_title(f"step {steps[i]}", color="white", fontsize=9)
        if j == 0:
            ax.text(0.02, 0.02, f"window +-{half:g}", color="white", transform=ax.transAxes, fontsize=9)
plt.tight_layout()
fig.savefig(args.out, dpi=110, facecolor="black")
