"""Position error of bodies within `band` units of a tile seam (tick-0 positions, seams every `tile` units) vs the rest, per model, at several ticks.
Usage: python3 scripts/seam_error_split.py results/gen/gen_multiclump128.npz baseline tier0 --tile 32 --band 3"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("keys", nargs="+")
ap.add_argument("--tile", type=float, default=32.0)
ap.add_argument("--band", type=float, default=3.0)
a = ap.parse_args()
d = np.load(a.npz)
tp = d["truth_pos"]
p0 = tp[0] + float(d["span"]) / 2
dist = np.minimum((p0 % a.tile), a.tile - (p0 % a.tile)).min(1)
near = dist < a.band
print("bodies near a seam", int(near.sum()), "of", len(near))
for k in a.keys:
    e = np.linalg.norm(d[f"{k}_pos"] - tp, axis=-1)
    print(k, {t: (round(float(e[t][near].mean()), 3), round(float(e[t][~near].mean()), 3)) for t in (5, 10, 20, 50) if t < len(e)})
