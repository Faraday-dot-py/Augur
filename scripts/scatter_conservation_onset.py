"""Onset times and extremes from results/conservation/*.npz: tick where median |dE|/scale and |dL|/Lscale first exceed 0.1, bound fraction below 0.5*truth,
outside-grid fraction above 0.1, max |dv|/dt, frac>0.9c. Usage: python3 scripts/scatter_conservation_onset.py"""
import glob
import os

import numpy as np

for f in sorted(glob.glob("results/conservation/*.npz")):
    name = os.path.basename(f)[:-4]
    d = np.load(f)
    t = d["ticks"]
    n = d["m_snaps"].shape[2]
    out = [name]
    for s in [x for x in ("m", "t4", "t1") if f"{x}_K" in d.files]:
        K0, U0 = d[f"{s}_K"][0], d[f"{s}_U"][0]
        dE = np.abs(d[f"{s}_E"] - d[f"{s}_E"][0]) / (K0 + np.abs(U0))
        lsc = d["r0"] * np.sqrt(2 * n * np.maximum(K0, 1e-12))
        dL = np.abs(d[f"{s}_L"] - d[f"{s}_L"][0]) / lsc

        def first(a, thr):
            m = np.median(a, axis=1) > thr
            return int(t[m.argmax()]) if m.any() else -1

        out.append(f"{s}: dE>0.1 @{first(dE, 0.1)}, dL>0.1 @{first(dL, 0.1)}, out>0.1 @{first(d[f'{s}_outgrid'], 0.1)}, amax {float(d[f'{s}_amax_all'].max()):.3g}, "
                   f"f90max {float(d[f'{s}_frac90'].max()):.3g}, vmax {float(d[f'{s}_vmax_all'].max()):.4g}")
    mb, tb = np.median(d["m_bound"], 1), np.median(d["t4_bound"], 1)
    m = mb < 0.5 * tb
    out.append(f"bound<0.5*truth @{int(t[m.argmax()]) if m.any() else -1}")
    print(" | ".join(out))
