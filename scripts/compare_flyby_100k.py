"""Model vs truth numbers from gravity_flyby_100k.py: energy/|P| at common diag ticks, radial quantiles about the median, and density-map correlation on a fixed 1000-unit window at several times.

Usage: python3 scripts/compare_flyby_100k.py results/flyby_100k.npz
"""
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

d = np.load(sys.argv[1])
m, t = d["model"], d["truth"]
dt, rec = float(d["dt"]), int(d["record"])
md, td, mt = d["model_diag"], d["truth_diag"], d["model_diag_t"]
print("tick  E_truth  E_model  |P|_truth  |P|_model")
for i, k in enumerate(d["truth_diag_t"]):
    if i % 4 == 0 or i == len(td) - 1:
        j = int(np.where(mt == k)[0][0])
        print(f"{k} {td[i, 0]:.5g} {md[j, 0]:.5g} {td[i, 1]:.4g} {md[j, 1]:.4g}")
edges = [np.linspace(a, a + 1000, 251) for a in (0, 0)]
print("t  r25/50/75 truth | model  density corr (window 1000 about truth median)")
for i in range(0, len(t), 25):
    c = np.median(t[i], axis=0)
    q = [np.percentile(np.linalg.norm(x[i] - c, axis=1), [25, 50, 75]).round(0) for x in (t, m)]
    e = [np.linspace(c[k] - 500, c[k] + 500, 251) for k in range(2)]
    im = [gaussian_filter(np.histogram2d(x[i][:, 0], x[i][:, 1], bins=e)[0], 1.5).ravel() for x in (t, m)]
    print(f"{i * rec * dt:.0f} {q[0]} | {q[1]}  {np.corrcoef(im[0], im[1])[0, 1]:.3f}")
