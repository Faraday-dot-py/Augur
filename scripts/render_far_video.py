"""Renders results/farvideo_*.npz (scripts/scatter_far_video.py) to an mp4: exact | tiled+far with the shipped kernel | tiled+far with the fixed kernel,
tile seams drawn, bodies coloured by position error vs exact (streams frames; ~5k bodies).

Usage: python3 scripts/render_far_video.py results/farvideo_gauss.npz videos/scatter_far_kernel_gauss.mp4
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
ex, old, fx = d["exact"], d["old"], d["fixed"]
nt, tile = int(d["nt"]), float(d["tile"])
L = nt * tile
steps = ex.shape[0]
lim = L / 2

fig, ax = plt.subplots(1, 3, figsize=(18, 6.4), dpi=90)
fig.patch.set_facecolor("black")
titles = ["exact (fp64)", "tiled + far, shipped kernel", "tiled + far, fixed kernel"]
sc = []
for a, t, P in zip(ax, titles, (ex, old, fx)):
    a.set_facecolor("black")
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(t, color="w", fontsize=11)
    for s in range(1, nt):
        c = -L / 2 + s * tile
        a.axvline(c, color="w", lw=0.5, alpha=0.5)
        a.axhline(c, color="w", lw=0.5, alpha=0.5)
    a.add_patch(plt.Rectangle((-L / 2, -L / 2), L, L, fill=False, ec="#888", lw=1.0, ls="--"))
    sc.append(a.scatter(P[0, :, 0], P[0, :, 1], s=3, c=np.zeros(P.shape[1]), cmap="turbo", vmin=0, vmax=3, lw=0))
txt = fig.suptitle("", color="w", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))


def update(i):
    e = []
    for k, (s_, P) in enumerate(zip(sc, (ex, old, fx))):
        s_.set_offsets(P[i])
        err = np.linalg.norm(P[i] - ex[i], axis=-1)
        s_.set_array(err)
        e.append(err.mean())
    txt.set_text(f"tick {i}   mean position error vs exact: shipped kernel {e[1]:.3f}   fixed kernel {e[2]:.3f}   (colour = error, 0..3 units)")
    return sc


ani = animation.FuncAnimation(fig, update, frames=steps, blit=False)
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=15, bitrate=3500))
