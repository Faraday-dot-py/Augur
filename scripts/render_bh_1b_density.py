"""Density video (log scale) from results/scatter_bh_1b_hist.npz: left full box +-32, middle zoom +-4, right zoom +-1 (each recentred on the subsample median).

Usage: python3 scripts/render_bh_1b_density.py results/scatter_bh_1b_hist.npz videos/scatter_bh_1b_density.mp4
"""
import subprocess
import sys

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

data = np.load(sys.argv[1])
keys = ["full", "zoom4", "zoom1"]
arrs = {k: data[k] for k in keys}
n = int(data["n"])
norms = {k: LogNorm(vmin=1, vmax=max(2, arrs[k].max())) for k in keys}
cmap = matplotlib.colormaps["viridis"]
T = arrs["full"].shape[0]
size = 512
proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * size}x{size}",
                         "-r", "15", "-i", "-", "-pix_fmt", "yuv420p", "-crf", "18", sys.argv[2]], stdin=subprocess.PIPE)
for t in range(T):
    ims = []
    for k in keys:
        f = arrs[k][t].astype(np.float64)
        f = f.reshape(size, 1024 // size, size, 1024 // size).sum((1, 3))
        rgb = (cmap(norms[k](np.maximum(f, 0.5)))[..., :3] * 255).astype(np.uint8)
        ims.append(rgb[::-1])
    proc.stdin.write(np.concatenate(ims, 1).tobytes())
proc.stdin.close()
assert proc.wait() == 0
