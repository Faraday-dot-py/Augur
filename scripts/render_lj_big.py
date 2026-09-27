"""Density-field video for lj_scaling.py's big-N snapshot npz (pos, box; no per-particle speed at this
scale). Whole periodic box fills the frame, local density in log scale.

Usage: PYTHONPATH=. python3 scripts/render_lj_big.py results/lj_scaling_big_snaps.npz videos/lj_big.mp4 [res] [fps] [label]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

d = np.load(sys.argv[1])
res = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 20
label = sys.argv[5] if len(sys.argv) > 5 else "learned LJ model"
frames, box = d["pos"], d["box"]
n = frames.shape[1]
edges = [np.linspace(0, box[k], res + 1) for k in range(2)]


def image(i):
    h = np.histogram2d(frames[i][:, 0], frames[i][:, 1], bins=edges)[0].astype(np.float32)
    return gaussian_filter(h, 1.0).T


vmax = np.log1p(np.percentile(image(len(frames) // 2), 99.9))
fig = plt.figure(figsize=(res / 100, res / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
im = ax.imshow(np.zeros((res, res), dtype=np.float32), origin="lower", cmap="inferno", vmin=0, vmax=vmax)
text = ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=16)


def update(i):
    im.set_data(np.log1p(image(i)))
    text.set_text(f"{n:,} particles ({label}), frame {i}")


ani = animation.FuncAnimation(fig, update, frames=len(frames))
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=fps, bitrate=10000), savefig_kwargs={"facecolor": "black"})
