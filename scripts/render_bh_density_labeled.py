"""Labeled three-panel density video (full box, zoom x8, zoom x32; log colour per panel) from a scatter_bh_1b.py --hist npz. Streams frames to ffmpeg.

Usage: python3 scripts/render_bh_density_labeled.py results/scatter_bh_10b_hist.npz videos/out.mp4 --title "10 billion bodies, learned gravity field" [--fps 12] [--frames N]
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
ap.add_argument("--frames", type=int, default=10 ** 9)
ap.add_argument("--dt", type=float, default=0.1)
args = ap.parse_args()

d = np.load(args.hist)
keys = ["full", "zoom4", "zoom1"]
arrs = {k: d[k] for k in keys}
nf = min(arrs["full"].shape[0], args.frames)
n = int(d["n"])
box, zh = float(d["box_half"]), [float(v) for v in d["zoom_half"]]
caps = [f"full box ({2 * box:g} units wide)", f"zoom x{box / zh[0]:g} ({2 * zh[0]:g} units, centred on the cluster)", f"zoom x{box / zh[1]:g} ({2 * zh[1]:g} units)"]
norms = {k: LogNorm(vmin=1, vmax=max(2, arrs[k].max())) for k in keys}
cmap = matplotlib.colormaps["inferno"]
fig = plt.figure(figsize=(30.72, 11.2), dpi=100, facecolor="black")
axs = [fig.add_axes([i / 3, 0.0, 1 / 3, 0.9]) for i in range(3)]
ims = []
for a, c in zip(axs, caps):
    a.axis("off")
    ims.append(a.imshow(np.zeros((1024, 1024, 3), dtype=np.uint8), origin="lower"))
    a.text(0.5, 0.02, c, color="white", transform=a.transAxes, ha="center", fontsize=22)
title = fig.text(0.5, 0.95, "", color="white", ha="center", va="center", fontsize=32)
w, h = fig.canvas.get_width_height()
proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
                         "-pix_fmt", "yuv420p", "-crf", "16", args.out], stdin=subprocess.PIPE)
for t in range(nf):
    for im, k in zip(ims, keys):
        f = np.maximum(arrs[k][t].astype(np.float64), 0.5)
        im.set_data((cmap(norms[k](f))[..., :3] * 255).astype(np.uint8))
    title.set_text(f"{args.title}   |   {n:,} bodies   |   simulated time {t * args.dt:.1f}")
    fig.canvas.draw()
    proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
proc.stdin.close()
assert proc.wait() == 0
