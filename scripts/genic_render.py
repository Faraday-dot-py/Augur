"""Density grid (log counts, shared box per IC from the exact run) of exact vs adaptive / geo_audit / mesh / noise-floor exact rollouts.

Usage: python3 scripts/genic_render.py <ic> <out.png>   (reads results/rollout_genic_r50k_<ic>_<run>_snaps.npz)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ic, out = sys.argv[1], sys.argv[2]
runs = ["exact", "noise", "adaptive", "geoaudit", "mesh"]
want = [0, 100, 300, 600, 1000]
data = {r: np.load(f"results/rollout_genic_r50k_{ic}_{r}_snaps.npz") for r in runs}
ref = data["exact"]
allp = ref["pos"]
c = np.median(allp[0], 0)
half = 1.05 * np.quantile(np.linalg.norm(allp[-1] - np.median(allp[-1], 0), axis=1), 0.99)
half = max(half, 1.05 * np.quantile(np.linalg.norm(allp[0] - c, axis=1), 0.99))
fig, ax = plt.subplots(len(runs), len(want), figsize=(2.6 * len(want), 2.6 * len(runs)))
for i, r in enumerate(runs):
    d = data[r]
    for j, s in enumerate(want):
        p = d["pos"][list(d["steps"]).index(s)]
        h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=128, range=[[c[0] - half, c[0] + half], [c[1] - half, c[1] + half]])
        ax[i, j].imshow(np.maximum(h.T, 0.5), origin="lower", cmap="viridis", norm=LogNorm(vmin=0.5, vmax=max(h.max(), 2)))
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(f"step {s}")
        if j == 0:
            ax[i, j].set_ylabel(r)
plt.tight_layout()
plt.savefig(out, dpi=80)
