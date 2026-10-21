"""Truth | single grid | tiled video and still grid from escape_roll.py npz files (one frame in memory at a time; inputs ~40 MB each).
Usage: python3 scripts/render_escape.py DIR truth S1 T8 videos/out.mp4 videos/out_grid.png [half] [fps]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

d, names, out, grid = sys.argv[1], sys.argv[2:5], sys.argv[5], sys.argv[6]
half = float(sys.argv[7]) if len(sys.argv) > 7 else 80.0
fps = int(sys.argv[8]) if len(sys.argv) > 8 else 20
Z = [np.load(f"{d}/{n}.npz") for n in names]
vt, V = Z[0]["vid_t"], [z["vid"] for z in Z]
labels = {"truth": "exact fp64", "S1": "single grid", "T8": "8x8 tiles + far"}
fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), dpi=80)
sc = []
for ax, n in zip(axs, names):
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_title(n if n not in labels else labels[n], fontsize=9)
    ax.add_patch(Rectangle((-32, -32), 64, 64, fill=False, ec="tab:red", lw=1))
    sc.append(ax.scatter([], [], s=0.3, c="k", alpha=0.5))
txt = fig.suptitle("")
w = animation.FFMpegWriter(fps=fps)
with w.saving(fig, out, dpi=80):
    for i in range(len(vt)):
        for s, v in zip(sc, V):
            s.set_offsets(v[i])
        txt.set_text(f"tick {int(vt[i])}")
        w.grab_frame()
plt.close(fig)
ticks = [0, 100, 200, 300, 400, 500, 700, 1000]
idx = [int(np.argmin(np.abs(vt - t))) for t in ticks]
fig, axs = plt.subplots(3, len(ticks), figsize=(2.2 * len(ticks), 6.9), dpi=90)
for r, v in enumerate(V):
    for c, i in enumerate(idx):
        ax = axs[r, c]
        ax.scatter(v[i][:, 0], v[i][:, 1], s=0.2, c="k", alpha=0.5)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if r == 0:
            ax.set_title(f"tick {int(vt[i])}", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"row {r + 1}", fontsize=8)
fig.tight_layout()
fig.savefig(grid)
print("wrote", out, grid)
