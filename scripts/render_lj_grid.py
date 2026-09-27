"""Contact sheet of LJ frames: one row per npz, one column per frame index. Particles at true diameter, coloured
by speed with one fixed scale per row.

Usage: PYTHONPATH=. python3 scripts/render_lj_grid.py out.png 0,10,50,150,400,800 label=results/lj_X.npz [label=...]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np

out = sys.argv[1]
cols = [int(c) for c in sys.argv[2].split(",")]
rows = [a.split("=", 1) for a in sys.argv[3:]]
fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.2 * len(cols), 3.2 * len(rows)), facecolor="black", squeeze=False)
for r, (label, path) in enumerate(rows):
    d = np.load(path)
    vmax = np.percentile(np.linalg.norm(d["vel"][::10], axis=-1), 99)
    for c, i in enumerate(cols):
        ax = axes[r, c]
        box = d["box"]
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.set_aspect("equal")
        ax.axis("off")
        n = d["pos"].shape[1]
        col = EllipseCollection(np.full(n, 1.0), np.full(n, 1.0), np.zeros(n), units="xy", offsets=d["pos"][min(i, len(d["pos"]) - 1)],
                                offset_transform=ax.transData, cmap="turbo", clim=(0, vmax))
        col.set_array(np.linalg.norm(d["vel"][min(i, len(d["pos"]) - 1)], axis=-1))
        ax.add_collection(col)
        if r == 0:
            ax.set_title(f"frame {i}", color="white", fontsize=11)
        if c == 0:
            ax.text(-0.02, 0.5, label, color="white", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=11)
fig.subplots_adjust(left=0.04, right=0.995, top=0.96, bottom=0.005, wspace=0.02, hspace=0.02)
fig.savefig(out, dpi=90, facecolor="black")
