"""Streaming single-panel model video for a black-hole snapshot npz (orbit_bh.py), static camera:
centre and window are fixed from the full trajectory (median of frame 0, half-width from a late-frame
radius percentile), not tracked per frame. One frame image in memory at a time.

Usage: PYTHONPATH=. python3 scripts/render_bh_static.py results/orbit_<tag>_snaps.npz videos/out.mp4 [res] [blur_px] [half]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

d = np.load(sys.argv[1])
res = int(sys.argv[3]) if len(sys.argv) > 3 else 960
blur = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
frames, steps = d["pos"], d["steps"]
centre = np.median(frames[0], axis=0)
half = float(sys.argv[5]) if len(sys.argv) > 5 else max(3.5 * np.percentile(np.linalg.norm(frames[0] - centre, axis=1), 45), 20)
edges = [np.linspace(centre[k] - half, centre[k] + half, res + 1) for k in range(2)]


def image(i):
    h = np.histogram2d(frames[i][:, 0], frames[i][:, 1], bins=edges)[0].astype(np.float32)
    return gaussian_filter(h, blur).T / (2 * half / res) ** 2


peaks = [np.percentile(image(i), 99.95) for i in range(0, len(frames), max(1, len(frames) // 8))]
vmax = np.log1p(0.5 * max(peaks))
fig = plt.figure(figsize=(res / 100, res / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
im = ax.imshow(np.zeros((res, res), dtype=np.float32), origin="lower", cmap="inferno", vmin=0, vmax=vmax)
label = ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=16)


def update(i):
    im.set_data(np.log1p(image(i)))
    label.set_text(f"step {steps[i]}")
    return im, label


ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=20, bitrate=4000))
