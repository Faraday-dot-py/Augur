"""Two-panel figure from lj_scaling.py's json: wall-clock cost per particle vs N (O(N) check), and
intensive quantities (T, P, psi6) vs N at fixed density (tiling check).

Usage: PYTHONPATH=. python3 scripts/render_lj_scaling_plot.py results/lj_scaling.json out.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open(sys.argv[1]))
out = sys.argv[2]
timing, tiling = d["timing"], d["tiling"]
n = np.array([t["n"] for t in timing])
us = np.array([t["ms_per_particle_us"] for t in timing])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(n, us, "o-", color="C0")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("N particles")
ax1.set_ylabel("us per particle per force eval")
ax1.set_title(f"O(N) cost (flat/falling, not O(N) rising) -- correctness vs dense: {d['correctness_max_abs']:.1e}")

nt = np.array([t["n"] for t in tiling])
T = np.array([t["T"] for t in tiling])
P = np.array([t["P"] for t in tiling])
psi = np.array([t["psi6"] for t in tiling])
ax2b = ax2.twinx()
l1, = ax2.plot(nt, T, "o-", color="C1", label="T")
l2, = ax2.plot(nt, psi, "s-", color="C2", label="psi6")
l3, = ax2b.plot(nt, P, "^-", color="C3", label="P")
ax2.set_xscale("log")
ax2.set_xlabel("N particles (same density, tiled)")
ax2.set_ylabel("T, psi6")
ax2b.set_ylabel("P")
ax2.set_title(f"tiling: intensive quantities vs N (rho={d['rho']}, T0={d['temp']})")
ax2.legend(handles=[l1, l2, l3], fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=130)
