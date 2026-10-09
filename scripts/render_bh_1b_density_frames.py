"""Contact sheet of the log-density histograms in results/scatter_bh_1b_hist.npz-style files: rows = ticks, columns = full box +-32, zoom +-4, zoom +-1.

Usage: python3 scripts/render_bh_1b_density_frames.py results/scatter_bh_10b_hist.npz videos/scatter_bh_10b_density_frames.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

data = np.load(sys.argv[1])
keys = ["full", "zoom4", "zoom1"]
arrs = {k: data[k] for k in keys}
T = arrs["full"].shape[0]
ticks = [int(round(i)) for i in np.linspace(0, T - 1, 8)]
norms = {k: LogNorm(vmin=1, vmax=max(2, arrs[k].max())) for k in keys}
fig, axes = plt.subplots(len(ticks), 3, figsize=(9, 3 * len(ticks)))
for r, t in enumerate(ticks):
    for c, k in enumerate(keys):
        ax = axes[r, c]
        ax.imshow(np.maximum(arrs[k][t].astype(np.float64), 1e-9), norm=norms[k], cmap="viridis", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f"frame {t}")
        if r == 0:
            ax.set_title(k)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=70)
