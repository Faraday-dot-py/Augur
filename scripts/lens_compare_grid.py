"""Side-by-side frames: 1M-subsample lensed video (top) vs all-bodies lensed video (bottom). Usage: python3 scripts/lens_compare_grid.py out.png [frame ...]"""
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out, frames = sys.argv[1], [int(f) for f in sys.argv[2:]] or [0, 20, 40, 60, 100]


def grab(video, i):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=eq(n\\,{i})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1080, 1080, 3)


fig, axs = plt.subplots(2, len(frames), figsize=(3.2 * len(frames), 6.6), facecolor="black")
for j, i in enumerate(frames):
    for r, (v, name) in enumerate((("videos/scatter_bh_1b_lensed.mp4", "1M subsample"), ("videos/scatter_bh_1b_lensed_all.mp4", "all 1e9 bodies"))):
        axs[r, j].imshow(grab(v, i))
        axs[r, j].axis("off")
        axs[r, j].set_title(f"{name}, frame {i}", color="white", fontsize=9)
fig.tight_layout(pad=0.3)
fig.savefig(out, dpi=150, facecolor="black")
