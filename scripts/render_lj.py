"""Streaming video for lj_sim.py npz (pos, vel, box, stats). Particles drawn at true diameter (sigma), coloured by speed
with one fixed scale per file; periodic box fills the frame. Optional second npz gives a side-by-side panel.

Usage: PYTHONPATH=. python3 scripts/render_lj.py results/lj_X.npz videos/lj_X.mp4 [res] [fps] [other.npz] [label_a] [label_b]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np

res = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 30
paths = [sys.argv[1]] + ([sys.argv[5]] if len(sys.argv) > 5 else [])
labels = [sys.argv[6] if len(sys.argv) > 6 else "", sys.argv[7] if len(sys.argv) > 7 else ""]
data = [np.load(p) for p in paths]
nf = min(len(d["pos"]) for d in data)
vmax = np.percentile(np.linalg.norm(data[0]["vel"][::10], axis=-1), 99)
panels = len(data)
fig = plt.figure(figsize=(res * panels / 100, res / 100), dpi=100, facecolor="black")
cols, texts = [], []
for k, d in enumerate(data):
    ax = fig.add_axes([k / panels, 0, 1 / panels, 1])
    box = d["box"]
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_aspect("equal")
    ax.axis("off")
    n = d["pos"].shape[1]
    col = EllipseCollection(np.full(n, 1.0), np.full(n, 1.0), np.zeros(n), units="xy", offsets=d["pos"][0],
                            offset_transform=ax.transData, cmap="turbo", clim=(0, vmax))
    ax.add_collection(col)
    cols.append(col)
    texts.append(ax.text(0.02, 0.98, "", color="white", transform=ax.transAxes, va="top", fontsize=16))


def update(i):
    for k, d in enumerate(data):
        cols[k].set_offsets(d["pos"][i])
        cols[k].set_array(np.linalg.norm(d["vel"][i], axis=-1))
        st = d["stats"][i]
        texts[k].set_text(f"{labels[k] + ' ' if labels[k] else ''}N={d['pos'].shape[1]:,}  t={st[0] * float(d['dt']):.0f}  T={st[1]:.2f}  P={st[4]:.2f}")


ani = animation.FuncAnimation(fig, update, frames=nf)
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=fps, bitrate=8000), savefig_kwargs={"facecolor": "black"})
