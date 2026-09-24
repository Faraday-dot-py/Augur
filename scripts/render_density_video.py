"""Density-only heatmap video (log scale, gravity down) from results/tiled_video_*.npz.

Usage: python3 scripts/render_density_video.py results/tiled_video_10m.npz videos/tiled_10m_density.mp4
"""
import subprocess
import sys

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

data = np.load(sys.argv[1])
dens = data["density"].astype(float)
norm = LogNorm(vmin=1, vmax=dens.max())
cmap = matplotlib.colormaps["viridis"]
size = dens.shape[1] * 2
proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}",
                         "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", "-crf", "18", sys.argv[2]], stdin=subprocess.PIPE)
for f in dens:
    rgb = (cmap(norm(np.maximum(f, 0.5)))[..., :3] * 255).astype(np.uint8)
    rgb = rgb.repeat(2, axis=0).repeat(2, axis=1)
    proc.stdin.write(rgb.tobytes())
proc.stdin.close()
proc.wait()
