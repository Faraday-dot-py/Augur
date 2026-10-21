"""Renders results/seamvideo.npz (scripts/scatter_tiling_seamvideo.py) to an mp4: exact | single grid | 2x2 tiled, with tile seams
and halo bands drawn; tiled bodies are coloured by owning tile.

Usage: python3 scripts/render_tiling_seams.py results/seamvideo.npz videos/scatter_tiling_seams.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
ex, si, ti = d["exact"], d["single"], d["tiled"]
nt, tile, halo = int(d["nt"]), float(d["tile"]), float(d["halo"])
L = nt * tile
steps = ex.shape[0]
cols = np.array(plt.cm.Set1.colors)[np.arange(nt * nt) % 9] if nt > 2 else np.array(["#4cc9f0", "#f72585", "#ffd166", "#06d6a0"])
lim = 80


def owner(p):
    t = np.floor((p + L / 2) / tile).clip(0, nt - 1).astype(int)
    return t[:, 0] * nt + t[:, 1]


fig, ax = plt.subplots(1, 3, figsize=(18, 6.4), dpi=100)
fig.patch.set_facecolor("black")
titles = ["exact (fp64)", "single 128 grid", f"{nt}x{nt} tiles of {tile:g} (halo {halo:g})"]
sc = []
for a, t, P in zip(ax, titles, (ex, si, ti)):
    a.set_facecolor("black")
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([]); a.set_autoscale_on(False)
    a.set_title(t, color="w", fontsize=11)
    for s in range(1, nt):
        c = -L / 2 + s * tile
        a.axvline(c, color="w", lw=1.6 if nt <= 2 else 0.7, alpha=0.9 if nt <= 2 else 0.6); a.axhline(c, color="w", lw=1.6 if nt <= 2 else 0.7, alpha=0.9 if nt <= 2 else 0.6)
    a.add_patch(plt.Rectangle((-L / 2, -L / 2), L, L, fill=False, ec="#888", lw=1.0, ls="--"))
for s in range(1, nt):
    c = -L / 2 + s * tile
    if halo < tile / 2: ax[2].axvspan(c - halo, c + halo, color="w", alpha=0.10, lw=0)
    if halo < tile / 2: ax[2].axhspan(c - halo, c + halo, color="w", alpha=0.10, lw=0)
for a, P in zip(ax, (ex, si, ti)):
    sc.append(a.scatter(P[0, :, 0], P[0, :, 1], s=3 if nt <= 2 else 6, c="#cccccc", lw=0))
sc[2].set_color(cols[owner(ti[0])])
txt = fig.suptitle("", color="w", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))


def update(i):
    for s_, P in zip(sc, (ex, si, ti)):
        s_.set_offsets(P[i])
    sc[2].set_color(cols[owner(ti[i])])
    e1 = np.linalg.norm(ti[i] - ex[i], axis=-1).mean()
    e2 = np.linalg.norm(ti[i] - si[i], axis=-1).mean()
    txt.set_text(f"tick {i}   tiled-vs-exact err {e1:.3f}   tiled-vs-single {e2:.3f}   (tile halo {halo:g} units)")
    return sc


ani = animation.FuncAnimation(fig, update, frames=steps, blit=False)
ani.save(sys.argv[2], writer=animation.FFMpegWriter(fps=20, bitrate=4000))
