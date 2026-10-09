"""Side-by-side video of the scatter-field BH run: full-box density (no lens), force-magnitude map, far-field (FFT kernel) potential.

Inputs: results/<tag>_hist.npz (full-box histogram of all bodies) and results/<tag>_grids.npy (T, 4, 128, 128: density, |a|, far-field potential, net potential), both from
scripts/scatter_bh_1b.py --hist --grid-dump. Colour limits are fixed over all frames.

Usage: python3 scripts/render_bh_1b_panels.py results/scatter_bh_1b_panels videos/scatter_bh_1b_panels.mp4
"""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm

stem = sys.argv[1]
hist = np.load(stem + "_hist.npz")
full = hist["full"].astype(np.float64)
grids = np.load(stem + "_grids.npy").astype(np.float64)
T = min(full.shape[0], grids.shape[0])
n = int(hist["n"])
half = float(hist["box_half"])
ext = [-half, half, -half, half]
acc = np.maximum(grids[:, 1], 1e-9)
phi = grids[:, 2]
nd = LogNorm(vmin=1, vmax=max(2, full[:T].max()))
na = LogNorm(vmin=max(acc[:T].max() * 1e-4, 1e-9), vmax=acc[:T].max())
pl, ph = phi[:T].min(), phi[:T].max()

fig = plt.figure(figsize=(16.2, 5.6), dpi=100)
canvas = FigureCanvasAgg(fig)
axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.04, wspace=0.08)
W, H = (int(v) for v in fig.get_size_inches() * fig.dpi)
proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "15", "-i", "-",
                         "-pix_fmt", "yuv420p", "-crf", "18", sys.argv[2]], stdin=subprocess.PIPE)
for t in range(T):
    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].imshow(np.maximum(full[t], 1e-9), norm=nd, cmap="viridis", origin="lower", extent=ext, interpolation="nearest")
    axes[0].set_title(f"all {n:.0e} bodies, density (log), tick {t}")
    axes[1].imshow(acc[t], norm=na, cmap="magma", origin="lower", extent=ext, interpolation="bilinear")
    axes[1].set_title("model force |a| on the 128 grid (log)")
    axes[2].imshow(phi[t], vmin=pl, vmax=ph, cmap="cividis", origin="lower", extent=ext, interpolation="bilinear")
    axes[2].set_title("far-field potential (FFT kernel term)")
    canvas.draw()
    proc.stdin.write(canvas.buffer_rgba().tobytes())
proc.stdin.close()
proc.wait()
