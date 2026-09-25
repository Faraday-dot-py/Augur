"""Density (relative to the mean) and mean-speed maps of the 1B-body run as a video.

Usage: PYTHONPATH=. python3 scripts/render_gravity_1b.py results/gravity_1b.npz videos/gravity_1b.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
dens, spd, steps = d["density"], d["speed"], d["steps"]
dt, bodies, spread = float(d["dt"]), int(d["bodies"]), float(d["spread"])
mean_d = dens[0][dens[0] > 0].mean()
rel = dens / mean_d
lim = max(0.05, float(np.percentile(np.abs(rel - 1), 99.5)))
fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), dpi=90)
im0 = axes[0].imshow(rel[0].T, origin="lower", cmap="RdBu_r", vmin=1 - lim, vmax=1 + lim)
im1 = axes[1].imshow(spd[0].T, origin="lower", cmap="viridis", vmin=0, vmax=np.percentile(spd, 99.5))
axes[0].set_title("density / mean")
axes[1].set_title("mean speed")
for ax, im in zip(axes, (im0, im1)):
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
title = fig.suptitle("")
fig.tight_layout(rect=(0, 0, 1, 0.94))


def update(i):
    im0.set_data(rel[i].T)
    im1.set_data(spd[i].T)
    title.set_text(f"{bodies:,} bodies, step {steps[i]}, t = {steps[i] * dt:.0f}, window {2.5 * spread:.0f} cells wide, "
                   f"KE {d['ke'][i]:.3g}, |P| {d['momentum'][i]:.2g}")


ani = animation.FuncAnimation(fig, update, frames=len(steps))
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=10, bitrate=3000))
