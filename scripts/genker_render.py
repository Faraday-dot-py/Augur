"""genker: side-by-side density frames (log counts, common extent per column) of exact vs candidate providers from results/genker_<tag>_<prov>_snaps.npz.

Usage: python3 scripts/genker_render.py --tag lj --provs exact,adaptive_own,adaptive_transfer --steps 0 100 300 600 1000 --out videos/genker_lj_grid.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--provs", default="exact,adaptive_own")
ap.add_argument("--steps", type=int, nargs="+", default=[0, 100, 300, 600, 1000])
ap.add_argument("--bins", type=int, default=160)
ap.add_argument("--out", required=True)
a = ap.parse_args()
provs = a.provs.split(",")
data = {p: np.load(f"results/genker_{a.tag}_{p}_snaps.npz") for p in provs}
fig, axes = plt.subplots(len(provs), len(a.steps), figsize=(3 * len(a.steps), 3 * len(provs)), squeeze=False)
for j, s in enumerate(a.steps):
    ref = data[provs[0]]["pos"][list(data[provs[0]]["steps"]).index(s)]
    c = np.median(ref, 0)
    half = np.quantile(np.linalg.norm(ref - c, axis=1), 0.99) * 1.1
    for i, p in enumerate(provs):
        pos = data[p]["pos"][list(data[p]["steps"]).index(s)]
        h, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=a.bins, range=[[c[0] - half, c[0] + half], [c[1] - half, c[1] + half]])
        axes[i, j].imshow(np.log1p(h.T), origin="lower", cmap="viridis")
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if i == 0:
            axes[i, j].set_title(f"step {s}")
        if j == 0:
            axes[i, j].set_ylabel(p)
plt.tight_layout()
plt.savefig(a.out, dpi=80)
