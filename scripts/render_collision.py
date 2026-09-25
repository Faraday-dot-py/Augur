"""Truth vs model density video from gravity_collision.py: additive glow on black, brightness ~ log(1 + bodies per pixel).

Usage: PYTHONPATH=. python3 scripts/render_collision.py results/collision_10k.npz videos/collision_10k.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

d = np.load(sys.argv[1])
truth, model = d["truth"], d["model"]
dt, record, n = float(d["dt"]), int(d["record"]), int(d["bodies"])
res, blur = 400, 1.3
pts = truth.reshape(-1, 2)
lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
pad = 0.15 * (hi - lo)
lo, hi = lo - pad, hi + pad
side = float((hi - lo).max())
mid = (lo + hi) / 2
edges = [np.linspace(mid[k] - side / 2, mid[k] + side / 2, res + 1) for k in range(2)]


def image(p):
    h = np.histogram2d(p[:, 0], p[:, 1], bins=edges)[0]
    return gaussian_filter(h, blur).T


imgs = [(image(truth[i]), image(model[i])) for i in range(len(truth))]
vmax = np.log1p(max(np.percentile(np.stack([a for a, _ in imgs]), 99.9), 1e-3))
fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), dpi=90, facecolor="black")
ims = []
for ax, title in zip(axes, ("truth (exact all-pairs)", "model (cutoff 4 + far field)")):
    ims.append(ax.imshow(np.zeros((res, res)), origin="lower", cmap="inferno", vmin=0, vmax=vmax))
    ax.set_title(title, color="white", fontsize=11)
    ax.axis("off")
label = fig.suptitle("", color="white")
fig.tight_layout(rect=(0, 0, 1, 0.94))


def update(i):
    for im, a in zip(ims, imgs[i]):
        im.set_data(np.log1p(a))
    label.set_text(f"{n:,} bodies, two-cluster collision, t = {i * record * dt:.0f}")


ani = animation.FuncAnimation(fig, update, frames=len(imgs))
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=30, bitrate=4000), savefig_kwargs={"facecolor": "black"})
