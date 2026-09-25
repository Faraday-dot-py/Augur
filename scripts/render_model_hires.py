"""High-resolution single-panel video of the model rollout from gravity_collision.py: additive glow on black, brightness ~ log(1 + bodies per pixel).
Frames are computed one at a time (peak memory ~ one res x res image plus the position array).

Usage: PYTHONPATH=. python3 scripts/render_model_hires.py results/flyby_10k.npz videos/flyby_10k_model_hires.mp4 [res] [blur]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

d = np.load(sys.argv[1])
model = d["model"]
dt, record, n = float(d["dt"]), int(d["record"]), int(d["bodies"])
res = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
blur = float(sys.argv[4]) if len(sys.argv) > 4 else 2.2
pts = model.reshape(-1, 2)
lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
pad = 0.15 * (hi - lo)
lo, hi = lo - pad, hi + pad
side = float((hi - lo).max())
mid = (lo + hi) / 2
edges = [np.linspace(mid[k] - side / 2, mid[k] + side / 2, res + 1) for k in range(2)]


def image(p):
    h = np.histogram2d(p[:, 0], p[:, 1], bins=edges)[0].astype(np.float32)
    return gaussian_filter(h, blur).T


peaks = [np.percentile(image(model[i]), 99.9) for i in range(0, len(model), 10)]
vmax = np.log1p(max(max(peaks), 1e-3))
fig = plt.figure(figsize=(res / 100, res / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
im = ax.imshow(np.zeros((res, res), dtype=np.float32), origin="lower", cmap="inferno", vmin=0, vmax=vmax)
label = ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=14)


def update(i):
    im.set_data(np.log1p(image(model[i])))
    label.set_text(f"{n:,} bodies (model), t = {i * record * dt:.0f}")


ani = animation.FuncAnimation(fig, update, frames=len(model))
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=30, bitrate=10000), savefig_kwargs={"facecolor": "black"})
