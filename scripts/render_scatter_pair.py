"""Streaming side-by-side video (model | truth) of a scatter_bh.py snapshot npz, static camera, one frame in memory at a time.

Usage: PYTHONPATH=. python3 scripts/render_scatter_pair.py results/<tag>_snaps.npz videos/out.mp4 [half] [fps]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
out = sys.argv[2]
half = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
fps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
pos, truth = d["pos"], d["truth_pos"]
c = np.median(truth[0], axis=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 5.2), dpi=100)
sc = []
for ax, title in zip(axs, ("model", "truth")):
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")
    ax.set_title(title)
    sc.append(ax.scatter([], [], s=6, c="k"))
txt = fig.suptitle("")
w = animation.FFMpegWriter(fps=fps)
with w.saving(fig, out, dpi=100):
    for t in range(len(pos)):
        sc[0].set_offsets(pos[t])
        sc[1].set_offsets(truth[t])
        txt.set_text(f"step {t}")
        w.grab_frame()
