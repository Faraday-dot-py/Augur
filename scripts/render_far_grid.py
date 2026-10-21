"""Frame grid for the unbiased review of the wide-scene far-kernel video: rows = three runs (labels A, B, C, order shuffled and hidden in the
legend file), columns = ticks. Usage: python3 scripts/render_far_grid.py results/farvideo_gauss.npz videos/scatter_far_kernel_gauss_grid.png [legend.txt]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
runs = {"exact": d["exact"], "shipped": d["old"], "fixed": d["fixed"]}
rng = np.random.default_rng(4738)
names = list(runs)
rng.shuffle(names)
labels = "ABC"
ticks = [0, 5, 10, 20, 50, 100]
nt, tile = int(d["nt"]), float(d["tile"])
L = nt * tile
fig, ax = plt.subplots(3, len(ticks), figsize=(3.0 * len(ticks), 9.2), dpi=80)
for i, nm in enumerate(names):
    for j, t in enumerate(ticks):
        a = ax[i, j]
        P = runs[nm][t]
        a.scatter(P[:, 0], P[:, 1], s=1.5, c="k", lw=0)
        a.set_xlim(-L / 2, L / 2)
        a.set_ylim(-L / 2, L / 2)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        if i == 0:
            a.set_title(f"tick {t}", fontsize=10)
        if j == 0:
            a.set_ylabel(f"run {labels[i]}", fontsize=11)
plt.tight_layout()
plt.savefig(sys.argv[2])
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write("\n".join(f"run {labels[i]} = {nm}" for i, nm in enumerate(names)) + "\n")
