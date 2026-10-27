"""Error and energy curves per arm and tick from the history npz written by scripts/exp12_seam_freeroll.py --hist-out; plus share of particles that ever changed owner.
Usage: python3 scripts/exp12_seam_freeroll_curves.py hist_all.npz"""
import sys

import numpy as np

d = np.load(sys.argv[1])
arms = sorted({k.split("_")[0] for k in d.files if k.endswith("_P") and not k.startswith("truth")})
for arm in arms:
    out = {t: [] for t in (5, 10, 20, 50, 100)}
    dE = []
    ever = []
    for sd in (9100, 9200, 9300):
        P, T, mk = d[f"{arm}_{sd}_P"], d[f"truth_{sd}_P"], d[f"mask_{sd}"]
        e = np.linalg.norm(P - T, axis=-1)
        for t in out:
            out[t].append(((e[:, t] * mk).sum(1) / mk.sum(1)).mean())
        E = d[f"{arm}_{sd}_E"]
        dE.append(np.abs((E[:, 100] - E[:, 0]) / E[:, 0]).mean())
        own = d[f"{arm}_{sd}_own"]
        ch = (own[:, 1:] != own[:, :-1]) & (mk[:, None] > 0)
        ever.append(ch.any(1).sum() / mk.sum())
    print(arm, {t: round(float(np.mean(v)), 3) for t, v in out.items()}, "dE@100", round(float(np.mean(dE)), 4), "frac ever changed owner", round(float(np.mean(ever)), 3))
