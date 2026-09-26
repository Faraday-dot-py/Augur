"""Compare two nbody_rollout snapshot npz (pos, steps): per common step, density-map correlation, radial-quantile ratios, COM offset.
Usage: python scripts/compare_snaps.py a.npz b.npz out.json [bins]
"""
import json
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
bins = int(sys.argv[4]) if len(sys.argv) > 4 else 128
sa, sb = list(a["steps"]), list(b["steps"])
pa, pb = a["pos"], b["pos"]
out = []
for s in sorted(set(sa) & set(sb)):
    if s % 250:
        continue
    p, q = pa[sa.index(s)].astype(np.float64), pb[sb.index(s)].astype(np.float64)
    cq = np.median(q, 0)
    rq_, rp_ = np.linalg.norm(q - cq, axis=1), np.linalg.norm(p - np.median(p, 0), axis=1)
    half = np.percentile(rq_, 99) * 1.05
    e = [np.linspace(cq[k] - half, cq[k] + half, bins + 1) for k in range(2)]
    hp, hq = np.histogram2d(p[:, 0], p[:, 1], bins=e)[0].ravel(), np.histogram2d(q[:, 0], q[:, 1], bins=e)[0].ravel()
    qs = (0.25, 0.5, 0.75, 0.9, 0.99)
    out.append({"step": int(s), "density_corr": float(np.corrcoef(hp, hq)[0, 1]), "rq_ratio": [float(np.quantile(rp_, x) / np.quantile(rq_, x)) for x in qs],
                "median_offset": float(np.linalg.norm(np.median(p, 0) - cq)), "dx_mean": float(np.linalg.norm(p - q, axis=1).mean())})
    print(out[-1], flush=True)
json.dump(out, open(sys.argv[3], "w"))
