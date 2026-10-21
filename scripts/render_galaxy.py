"""Side-by-side galaxy density video: columns = exact all-pairs truth (N=1e5) | scatter-field model (N=1e5) | scatter-field model (N=1e9, every body binned); rows = wide window (+-24) and bulge window (+-8).
Colour = log10 of surface density (fraction of bodies per unit area), same limits in every panel. The 1e5 panels are binned 4x4 (192^2 bins) so a bin holds a usable number of bodies; the 1e9 panel is drawn at 768^2. Frames are streamed (one at a time from memmaps).
Usage: python3 scripts/render_galaxy.py <dir with g1b g_mesh_100000 g_truth_100000> videos/scatter_galaxy_1e9.mp4"""
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

root, outp = sys.argv[1], sys.argv[2]
runs = [("exact truth, N=1e5", "g_truth_100000", 100000), ("scatter-field model, N=1e5", "g_mesh_100000", 100000), ("scatter-field model, N=1e9", "g1b", 10 ** 9)]
HW = (24.0, 8.0)
QS = 3000.0
data = []
for title, tag, n in runs:
    A = np.load(f"{root}/{tag}/hist_A.npy", mmap_mode="r")
    B = np.load(f"{root}/{tag}/hist_B.npy", mmap_mode="r")
    diag = json.load(open(f"{root}/{tag}/diag.json"))
    meta = json.load(open(f"{root}/{tag}/meta.json"))
    data.append((title, n, (A, B), diag, meta["args"]["rec_every"]))
T = min(d[2][0].shape[0] for d in data)
for d in data:
    last = d[3][-1]["tick"]
    T = min(T, last // d[4] + 1)


def dens(q, n, hw, rebin):
    c = np.expm1(q.astype(np.float64) / QS)
    res = c.shape[0]
    if rebin > 1:
        c = c.reshape(res // rebin, rebin, res // rebin, rebin).sum((1, 3))
        res //= rebin
    area = (2 * hw / res) ** 2
    return np.log10(np.maximum(c, 0.3) / n / area)


vmax = [max(dens(data[2][2][w][t], data[2][1], HW[w], 1).max() for t in range(0, T, 10)) for w in range(2)]
fig = plt.figure(figsize=(15.0, 10.4), dpi=100)
canvas = FigureCanvasAgg(fig)
axes = [[fig.add_subplot(2, 3, 3 * r + c + 1) for c in range(3)] for r in range(2)]
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.03, hspace=0.06)
W, H = (int(v) for v in fig.get_size_inches() * fig.dpi)
proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "15", "-i", "-", "-pix_fmt", "yuv420p", "-crf", "20", outp],
                        stdin=subprocess.PIPE)
for t in range(T):
    for c, (title, n, hists, diag, rec) in enumerate(data):
        tick = t * rec
        dd = [d for d in diag if d["tick"] <= tick][-1]
        for r in range(2):
            ax = axes[r][c]
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            rebin = 4 if n < 10 ** 6 else 1
            img = dens(hists[r][t], n, HW[r], rebin)
            ax.imshow(img, origin="lower", cmap="inferno", vmin=vmax[r] - 5.0, vmax=vmax[r], interpolation="nearest" if rebin > 1 else "bilinear", extent=[-HW[r], HW[r], -HW[r], HW[r]])
        axes[0][c].set_title(f"{title}\ntick {tick}   bound {dd['bound']:.3f}   dE {dd['dE']:+.3f}", fontsize=11)
    canvas.draw()
    proc.stdin.write(canvas.buffer_rgba().tobytes())
proc.stdin.close()
assert proc.wait() == 0
