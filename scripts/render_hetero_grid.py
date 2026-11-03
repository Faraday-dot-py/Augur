"""Diagnostic grid for hetero_tier_checks phase d: rows = exact / single / 2x2 tiled / 4x4 tiled, columns = ticks; bodies coloured by position error vs exact.
Usage: python scripts/render_hetero_grid.py results/hetero/hetero_d_traj.npz videos/hetero_seam_grid.png"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
ticks = (0, 10, 20, 30, 40)
rows = [("exact", "exact"), ("single", "single grid"), ("g16_fx_far", "tiled 2x2 (64)"), ("t32_g16_fx_far", "tiled 4x4 (32)")]
fig, ax = plt.subplots(len(rows), len(ticks), figsize=(3 * len(ticks), 3 * len(rows)))
ex = d["exact"]
for i, (k, lab) in enumerate(rows):
    for j, t in enumerate(ticks):
        a = ax[i, j]
        P = d[k][t]
        err = np.linalg.norm(d[k][t] - ex[t], axis=1)
        a.scatter(P[:, 0], P[:, 1], s=3, c=err, cmap="viridis", vmin=0, vmax=3)
        for s in (-32, 0, 32):
            a.axvline(s, color="r", lw=0.4)
            a.axhline(s, color="r", lw=0.4)
        a.set_xlim(-40, 40)
        a.set_ylim(-40, 40)
        a.set_xticks([])
        a.set_yticks([])
        if i == 0:
            a.set_title(f"tick {t}")
        if j == 0:
            a.set_ylabel(lab)
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=80)
