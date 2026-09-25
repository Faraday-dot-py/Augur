"""Frame grid (density/mean, mean speed) at selected steps of a gravity_1b.py run, for unbiased review.

Usage: PYTHONPATH=. python3 scripts/render_gravity_grid.py results/x.npz out.png [step ...]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
want = [int(s) for s in sys.argv[3:]] or [0, 50, 100, 200, 300, 400, 500]
steps = list(d["steps"])
idx = [steps.index(s) for s in want if s in steps]
dens, spd = d["density"], d["speed"]
rel = dens / dens[0][dens[0] > 0].mean()
lim = max(0.05, float(np.percentile(np.abs(rel - 1), 99.5)))
fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6.4), dpi=80)
for c, i in enumerate(idx):
    axes[0, c].imshow(rel[i].T, origin="lower", cmap="RdBu_r", vmin=1 - lim, vmax=1 + lim)
    axes[1, c].imshow(spd[i].T, origin="lower", cmap="viridis", vmin=0, vmax=np.percentile(spd, 99.5))
    axes[0, c].set_title(f"step {steps[i]}")
    axes[0, c].axis("off"); axes[1, c].axis("off")
axes[0, 0].text(-0.05, 0.5, "density/mean", transform=axes[0, 0].transAxes, rotation=90, ha="right", va="center")
axes[1, 0].text(-0.05, 0.5, "mean speed", transform=axes[1, 0].transAxes, rotation=90, ha="right", va="center")
fig.tight_layout()
fig.savefig(sys.argv[2])
