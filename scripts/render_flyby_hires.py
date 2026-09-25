"""Streaming single-panel model video for large-N flyby npz (gravity_flyby_100k.py). Window follows a
time-smoothed centre (median position) with half-width 3.5x the median radius (floor 150), so structure fills the frame;
brightness is log(1 + density per unit area) with one fixed scale. One frame image in memory at a time.

Usage: PYTHONPATH=. python3 scripts/render_flyby_hires.py results/flyby_100k.npz videos/flyby_100k_model.mp4 [res] [blur_px] [key]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

d = np.load(sys.argv[1])
res = int(sys.argv[3]) if len(sys.argv) > 3 else 1440
blur = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
key = sys.argv[5] if len(sys.argv) > 5 else "model"
frames = d[key]
dt, record, n = float(d["dt"]), int(d["record"]), int(d["bodies"])
centre = np.stack([np.median(f, axis=0) for f in frames])
radius = np.array([np.percentile(np.linalg.norm(f - c, axis=1), 45) for f, c in zip(frames, centre)])
centre = gaussian_filter1d(centre, 4, axis=0, mode="nearest")
half = np.maximum(3.5 * gaussian_filter1d(radius, 4, mode="nearest"), 150)


def image(i):
    e = [np.linspace(centre[i, k] - half[i], centre[i, k] + half[i], res + 1) for k in range(2)]
    h = np.histogram2d(frames[i][:, 0], frames[i][:, 1], bins=e)[0].astype(np.float32)
    return gaussian_filter(h, blur).T / (2 * half[i] / res) ** 2


peaks = [np.percentile(image(i), 99.95) for i in range(0, len(frames), 8)]
vmax = np.log1p(0.5 * max(peaks))
fig = plt.figure(figsize=(res / 100, res / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
im = ax.imshow(np.zeros((res, res), dtype=np.float32), origin="lower", cmap="inferno", vmin=0, vmax=vmax)
label = ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=16)
scale = ax.text(0.02, 0.02, "", color="white", transform=ax.transAxes, va="bottom", fontsize=14)


def update(i):
    im.set_data(np.log1p(image(i)))
    label.set_text(f"{n:,} bodies ({key}), t = {i * record * dt:.0f}")
    scale.set_text(f"view {2 * half[i]:.0f} units")


ani = animation.FuncAnimation(fig, update, frames=len(frames))
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=30, bitrate=12000), savefig_kwargs={"facecolor": "black"})
