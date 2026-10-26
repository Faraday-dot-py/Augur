"""Four-tick figure (rows: ticks, columns: density, model force |a|, far-field potential) from a *_ticks.npz made by bh_edge_panels_extract.py, same layout and colour rules as
render_bh_1b_panels.py (colour limits fixed over the ticks shown).

Usage: python3 scripts/render_bh_panels_frames.py results/<tag>_ticks.npz videos/<name>.png
       python3 scripts/render_bh_panels_frames.py --compare old_ticks.npz new_ticks.npz videos/<name>.png   (rows: density, force with the old model, force with the fixed model; columns: ticks; one force colour scale from the fixed run)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

compare = sys.argv[1] == "--compare"
paths = sys.argv[2:-1] if compare else sys.argv[1:-1]
d = [np.load(p) for p in paths]
ext = [-float(d[0]["box_half"])] * 2 + [float(d[0]["box_half"])] * 2
ext = [ext[0], ext[2], ext[1], ext[3]]
n = float(d[0]["n"])
tk = d[0]["ticks"]
if not compare:
    full, g = d[0]["full"].astype(np.float64), d[0]["grids"].astype(np.float64)
    acc = np.maximum(g[:, 1], 1e-9)
    nd = LogNorm(vmin=1, vmax=max(2, full.max()))
    na = LogNorm(vmin=max(acc.max() * 1e-4, 1e-9), vmax=acc.max())
    fig, axes = plt.subplots(len(tk), 3, figsize=(8, 2.8 * len(tk)))
    for r, t in enumerate(tk):
        axes[r, 0].imshow(np.maximum(full[r], 1e-9), norm=nd, cmap="viridis", origin="lower", extent=ext, interpolation="nearest")
        axes[r, 0].set_title(f"all {n:.0e} bodies, density (log), tick {t}", fontsize=8)
        axes[r, 1].imshow(acc[r], norm=na, cmap="magma", origin="lower", extent=ext, interpolation="bilinear")
        axes[r, 1].set_title("model force |a| on the 128 grid (log)", fontsize=8)
        axes[r, 2].imshow(g[r, 2], vmin=g[:, 2].min(), vmax=g[:, 2].max(), cmap="cividis", origin="lower", extent=ext, interpolation="bilinear")
        axes[r, 2].set_title("far-field potential (FFT kernel term)", fontsize=8)
else:
    old, new = d
    full = new["full"].astype(np.float64)
    ao, an = np.maximum(old["grids"][:, 1].astype(np.float64), 1e-9), np.maximum(new["grids"][:, 1].astype(np.float64), 1e-9)
    nd = LogNorm(vmin=1, vmax=max(2, full.max()))
    na = LogNorm(vmin=max(an.max() * 1e-4, 1e-9), vmax=an.max())
    fig, axes = plt.subplots(3, len(tk), figsize=(2.8 * len(tk), 8.6))
    for c, t in enumerate(tk):
        axes[0, c].imshow(np.maximum(full[c], 1e-9), norm=nd, cmap="viridis", origin="lower", extent=ext, interpolation="nearest")
        axes[0, c].set_title(f"tick {t}", fontsize=9)
        axes[1, c].imshow(ao[c], norm=na, cmap="magma", origin="lower", extent=ext, interpolation="bilinear")
        axes[2, c].imshow(an[c], norm=na, cmap="magma", origin="lower", extent=ext, interpolation="bilinear")
    axes[0, 0].set_ylabel(f"density ({n:.0e} bodies, fixed model)", fontsize=8)
    axes[1, 0].set_ylabel("force |a|, old model (zero-padded)", fontsize=8)
    axes[2, 0].set_ylabel("force |a|, fixed model (extended)", fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(sys.argv[-1], dpi=100)
