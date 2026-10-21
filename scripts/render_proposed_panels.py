"""Labelled side-by-side density video of saved histograms, streamed frame by frame (RAM = the loaded npz arrays only).

bh:   inputs are scatter_bh_1b.py --hist npz files (full box +-32 and zoom +-4 recentred on the subsample median); one row per input, panels [full | zoom4], log density,
      colour range shared across rows per panel type.
disc: inputs are results/disc/<tag>/hist.npz (star-centred frames, hw 16); one panel per input, log density, shared colour range.

Usage: python3 scripts/render_proposed_panels.py bh out.mp4 old=results/bh1e6_old_hist.npz new=results/bh1e6_new_hist.npz [--fps 15]
"""
import argparse
import subprocess

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["bh", "disc"])
ap.add_argument("out")
ap.add_argument("inputs", nargs="+")
ap.add_argument("--fps", type=float, default=15)
ap.add_argument("--size", type=int, default=512)
args = ap.parse_args()
cmap = matplotlib.colormaps["viridis"]
S = args.size
labels, data = [], []
for item in args.inputs:
    lab, path = item.split("=", 1)
    labels.append(lab)
    data.append(np.load(path))


def shrink(f, size):
    k = f.shape[-1] // size
    return f.astype(np.float64).reshape(size, k, size, k).sum((1, 3))


if args.kind == "bh":
    keys = ["full", "zoom4"]
    arrs = [[d[k] for k in keys] for d in data]
    T = min(a[0].shape[0] for a in arrs)
    norms = [LogNorm(vmin=1, vmax=max(2, max(shrink(a[j][t], S).max() for a in arrs for t in range(0, T, 5)))) for j in range(2)]
else:
    arrs = [d["frames"] for d in data]
    T = min(a.shape[0] for a in arrs)
    every = int(data[0]["every"])
    vm = max(2, max(shrink(a[t], S).max() for a in arrs for t in range(0, T, 5)))
    norm = LogNorm(vmin=1, vmax=vm)
ncol = 2 if args.kind == "bh" else len(arrs)
nrow = len(arrs) if args.kind == "bh" else 1
proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ncol * S}x{nrow * S}", "-r", str(args.fps), "-i", "-",
                         "-pix_fmt", "yuv420p", "-crf", "18", args.out], stdin=subprocess.PIPE)
for t in range(T):
    rows = []
    if args.kind == "bh":
        for i, a in enumerate(arrs):
            ims = [(cmap(norms[j](np.maximum(shrink(a[j][t], S), 0.5)))[..., :3] * 255).astype(np.uint8)[::-1] for j in range(2)]
            rows.append(np.concatenate(ims, 1))
        rgb = np.concatenate(rows, 0)
        tags = [(0, i * S, f"{labels[i]}  tick {t}  full +-32") for i in range(nrow)] + [(S, i * S, "zoom +-4") for i in range(nrow)]
    else:
        ims = [(cmap(norm(np.maximum(shrink(a[t], S), 0.5)))[..., :3] * 255).astype(np.uint8)[::-1] for a in arrs]
        rgb = np.concatenate(ims, 1)
        tags = [(i * S, 0, f"{labels[i]}  tick {t * every}") for i in range(ncol)]
    img = Image.fromarray(np.ascontiguousarray(rgb))
    dr = ImageDraw.Draw(img)
    for x, y, s in tags:
        dr.text((x + 8, y + 8), s, fill=(255, 255, 255))
    proc.stdin.write(np.asarray(img).tobytes())
proc.stdin.close()
assert proc.wait() == 0
