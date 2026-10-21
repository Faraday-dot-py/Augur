"""Side-by-side video of two edge_longroll.py snapshot files (default model | fixed model), same outward-flow scene, grid box [-32,32]^2 and the edge band 28-32 shaded.
One frame in memory at a time. Usage: python3 scripts/render_edge_pair.py E.npz fixed.npz videos/out.mp4 [half] [fps] [labelA] [labelB]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
out = sys.argv[3]
half = float(sys.argv[4]) if len(sys.argv) > 4 else 56.0
fps = int(sys.argv[5]) if len(sys.argv) > 5 else 20
labels = (sys.argv[6] if len(sys.argv) > 6 else "default (zero-padded kernel gradient)", sys.argv[7] if len(sys.argv) > 7 else "edge-fixed (learned ghost closure)")
T = min(len(a["snap_t"]), len(b["snap_t"]))
fig, axs = plt.subplots(1, 2, figsize=(11, 5.8), dpi=100)
sc = []
for ax, lab in zip(axs, labels):
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_title(lab, fontsize=9)
    ax.add_patch(Rectangle((-32, -32), 64, 64, fill=False, ec="tab:red", lw=1.2))
    ax.add_patch(Rectangle((-28, -28), 56, 56, fill=False, ec="tab:red", lw=0.6, ls="--"))
    sc.append(ax.scatter([], [], s=0.4, c="k", alpha=0.6))
txt = fig.suptitle("")
w = animation.FFMpegWriter(fps=fps)
with w.saving(fig, out, dpi=100):
    for i in range(T):
        for s, d in zip(sc, (a, b)):
            s.set_offsets(d["snaps"][i])
        n = [int((np.abs(d["snaps"][i]).max(1) > 32).sum()) for d in (a, b)]
        txt.set_text(f"tick {int(a['snap_t'][i])}   bodies outside the box: {n[0]} | {n[1]} of {a['snaps'].shape[1]} plotted")
        w.grab_frame()
print("wrote", out, T, "frames")
