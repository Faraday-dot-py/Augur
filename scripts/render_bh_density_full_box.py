"""Full-box-only density video from a scatter_bh_1b.py --hist npz; --interp N linearly blends N-1 extra density frames between stored ticks. Streams frames to ffmpeg.

Usage: python3 scripts/render_bh_density_full_box.py results/showcase_bh_1e10_kpadext_hist.npz videos/out.mp4 --title "10 billion bodies, learned gravity field" --fps 120 --interp 10
"""
import argparse
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ap = argparse.ArgumentParser()
ap.add_argument("hist")
ap.add_argument("out")
ap.add_argument("--title", required=True)
ap.add_argument("--fps", type=float, default=12)
ap.add_argument("--interp", type=int, default=1)
ap.add_argument("--dt", type=float, default=0.1)
args = ap.parse_args()

d = np.load(args.hist)
arr = d["full"]
nt = arr.shape[0]
n = int(d["n"])
box = float(d["box_half"])
norm = LogNorm(vmin=1, vmax=max(2, arr.max()))
cmap = matplotlib.colormaps["inferno"]
fig = plt.figure(figsize=(10.8, 11.2), dpi=100, facecolor="black")
ax = fig.add_axes([0.0, 0.0, 1.0, 0.9])
ax.axis("off")
im = ax.imshow(np.zeros((1024, 1024, 3), dtype=np.uint8), origin="lower")
ax.text(0.5, 0.02, f"full box ({2 * box:g} units wide)", color="white", transform=ax.transAxes, ha="center", fontsize=22)
title = fig.text(0.5, 0.95, "", color="white", ha="center", va="center", fontsize=24, linespacing=1.4)
w, h = fig.canvas.get_width_height()
proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
                         "-pix_fmt", "yuv420p", "-crf", "16", args.out], stdin=subprocess.PIPE)
k = args.interp
total = (nt - 1) * k + 1
for i in range(total):
    t, a = divmod(i, k)
    f = arr[t].astype(np.float64)
    if a:
        f = (1 - a / k) * f + (a / k) * arr[t + 1].astype(np.float64)
    im.set_data((cmap(norm(np.maximum(f, 0.5)))[..., :3] * 255).astype(np.uint8))
    title.set_text(f"{args.title}\n{n:,} bodies   |   simulated time {i / k * args.dt:.1f}")
    fig.canvas.draw()
    proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
proc.stdin.close()
assert proc.wait() == 0
