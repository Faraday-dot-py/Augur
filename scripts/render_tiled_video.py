"""Renders results/tiled_video_*.npz (scripts/tiled_video.py) to an mp4:
density map | mean-speed map | boundary crop.

Usage: python3 scripts/render_tiled_video.py results/tiled_video_10m.npz videos/tiled_10m.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

data = np.load(sys.argv[1])
dens, spd, crop = data["density"], data["speed"].astype(np.float32), data["crop"]
frames = dens.shape[0]
grid, bins = float(data["grid"]), dens.shape[1]
bpx = grid / bins
dmax = np.percentile(dens[0], 99.9) * 1.5
smax = max(np.percentile(spd, 99.9), 1e-3)
boundary = float(data["boundary"])

fig, ax = plt.subplots(1, 3, figsize=(18, 6.4), dpi=100)
im0 = ax[0].imshow(dens[0].T, origin="lower", vmin=0, vmax=dmax, cmap="viridis")
im1 = ax[1].imshow(spd[0].T, origin="lower", vmin=0, vmax=smax, cmap="magma")
im2 = ax[2].imshow(crop[0].T, origin="lower", vmin=0, vmax=max(crop.max(), 1), cmap="gray")
for a in ax[:2]:
    for s in range(1, int(data["strips"])):
        a.axvline(s * grid / int(data["strips"]) / bpx, color="w", lw=0.5, ls=":")
ax[2].axvline(boundary - float(data["crop_lo"][0]), color="r", lw=0.5, ls=":")
ax[0].set_title("balls per bin"); ax[1].set_title("mean speed per bin"); ax[2].set_title("300x300 crop at strip boundary (1 px = 1 unit)")
title = fig.suptitle("")
fig.tight_layout()


def update(i):
    im0.set_data(dens[i].T); im1.set_data(spd[i].T); im2.set_data(crop[i].T)
    title.set_text(f"{int(data['balls']):,} balls, grid {int(grid):,}, tick {i}")
    return im0, im1, im2


ani = animation.FuncAnimation(fig, update, frames=frames)
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=15, bitrate=4000))
