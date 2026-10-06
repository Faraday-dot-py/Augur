"""Delete-the-star scene video: model force field (log-magnitude heatmap + arrows) beside the analytic truth, from scripts/delete_star_field.py output.

Usage: PYTHONPATH=. python3 scripts/render_delete_star_field.py results/delete_star_field_E.npz videos/out.mp4 [fps]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
xs, acc, truth, warm = d["xs"], d["acc"], d["truth"], int(d["warm"])
T = acc.shape[0]
h = float(xs.max() + (xs[1] - xs[0]) / 2)
ext = (-h, h, -h, h)
LO, HI = -3.0, 1.5
rings = [4.0, 8.0, 12.0]
ang = np.arange(4) * np.pi / 2


def lmag(a):
    return np.log10(np.linalg.norm(a, axis=-1) + 1e-9)


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), dpi=100)
ims, qs = [], []
for ax, title in zip(axs, ("model force field", "truth force field")):
    ax.set_title(title)
    ax.set_xlim(-h, h)
    ax.set_ylim(-h, h)
    ax.set_aspect("equal")
    ims.append(ax.imshow(np.zeros((len(xs), len(xs))), extent=ext, origin="lower", vmin=LO, vmax=HI, cmap="magma", interpolation="nearest"))
    X, Y = np.meshgrid(xs, xs)
    qs.append(ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(X), color="w", scale=30, width=0.003, pivot="mid"))
    for r in rings:
        ax.plot(r * np.cos(ang), r * np.sin(ang), "c.", ms=6)
    star, = ax.plot([0], [0], "*", c="gold", ms=18, mec="k")
    ax.star = star
cb = fig.colorbar(ims[0], ax=axs, shrink=0.8, label="log10 |acceleration|")
ttl = fig.suptitle("")
w = animation.FFMpegWriter(fps=fps)
with w.saving(fig, sys.argv[2], dpi=100):
    for t in range(T):
        on = t < warm
        for k, (a, ax) in enumerate(zip((acc[t], truth if on else np.zeros_like(truth)), axs)):
            ims[k].set_data(lmag(a))
            u = unit(a) * (np.linalg.norm(a, axis=-1, keepdims=True) > 1e-3)
            qs[k].set_UVC(u[..., 0], u[..., 1])
            ax.star.set_visible(on)
        ttl.set_text(f"step {t}: " + ("star present (mass 20)" if on else f"star deleted (+{t - warm})"))
        w.grab_frame()
    for _ in range(fps):
        w.grab_frame()
