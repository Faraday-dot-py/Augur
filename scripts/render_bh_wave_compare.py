"""Streamed side-by-side video of BH collapse arms from the 20000-body subsample in bh_wave_run npz files (one frame in RAM at a time).
Usage: python scripts/render_bh_wave_compare.py <data_dir> <scene> <out.mp4> arm1,arm2,...  (panels: log density, half-width 24 and 6 stacked)"""
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, scene, out, arms = sys.argv[1:5]
arms = arms.split(",")
zs = {a: np.load(f"{d}/{scene}_{a}.npz") for a in arms}
T = zs[arms[0]]["sub"].shape[0]
fig, ax = plt.subplots(2, len(arms), figsize=(4 * len(arms), 8), dpi=80)
fig.canvas.draw()
w, h = fig.canvas.get_width_height()
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", "15", "-i", "-", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], stdin=subprocess.PIPE)
for t in range(T):
    for j, a in enumerate(arms):
        s = zs[a]["sub"][t]
        c = s.mean(0) if False else np.median(s, 0)
        for i, hw in enumerate((24.0, 6.0)):
            ax[i, j].clear()
            H = np.histogram2d(s[:, 0] - c[0], s[:, 1] - c[1], bins=160, range=[[-hw, hw], [-hw, hw]])[0]
            ax[i, j].imshow(np.log1p(H).T, origin="lower", cmap="inferno", vmin=0, vmax=np.log1p(40 if hw > 10 else 12))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_title(f"{a} tick {t} (+-{hw:g})" if i == 0 else f"+-{hw:g}", fontsize=9)
    fig.canvas.draw()
    p.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
p.stdin.close()
p.wait()
