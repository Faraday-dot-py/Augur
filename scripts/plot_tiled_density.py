"""Density heatmaps (balls per bin) from results/tiled_video_*.npz at chosen ticks.
x is drawn top-to-bottom so gravity (+x) points down.

Usage: python3 scripts/plot_tiled_density.py results/tiled_video_10m.npz results/tiled_10m_density.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

data = np.load(sys.argv[1])
dens = data["density"].astype(float)
ticks = [0, 50, 100, 200, 300]
vmax = dens.max()
fig, axes = plt.subplots(1, len(ticks), figsize=(4 * len(ticks) + 1, 4.6), dpi=110, constrained_layout=True)
for ax, t in zip(axes, ticks):
    im = ax.imshow(np.maximum(dens[t], 0.5), origin="upper", norm=LogNorm(vmin=1, vmax=vmax), cmap="viridis")
    ax.set_title(f"tick {t}")
    ax.set_xlabel("y bin"); ax.set_ylabel("x bin (down = +x)")
fig.colorbar(im, ax=axes, label="balls per bin (log)", shrink=0.85)
fig.suptitle(f"{int(data['balls']):,} balls, grid {int(data['grid']):,}, {dens.shape[1]}x{dens.shape[2]} bins")
fig.savefig(sys.argv[2])
