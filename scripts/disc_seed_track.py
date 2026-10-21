"""Debug: per-seed r, speed, enclosed mass, E, L and core-centre drift over time from results/disc/<tag>."""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--every", type=int, default=10)
a = ap.parse_args()
meta = json.load(open(f"{a.dir}/meta.json"))
P = np.load(f"{a.dir}/snaps_pos.npy")
V = np.load(f"{a.dir}/snaps_vel.npy")
w = np.load(f"{a.dir}/sub_wt.npy")
sub = np.load(f"{a.dir}/sub_idx.npy")
core = sub < meta["ncore"]
seeds = np.nonzero(sub >= meta["n_reg"])[0]
for j in range(0, P.shape[0], a.every):
    c = np.median(P[j, core], axis=0)
    cv = np.median(V[j, core], axis=0)
    row = [f"t{j * meta['snap_every']} ctr({c[0]:.2f},{c[1]:.2f}) cv({cv[0]:.2f},{cv[1]:.2f})"]
    for s in seeds:
        rel, vr = P[j, s] - c, V[j, s] - cv
        r = np.linalg.norm(rel)
        mu = w[np.linalg.norm(P[j] - c, axis=1) < r].sum()
        E = 0.5 * (vr ** 2).sum() - mu / np.sqrt(r * r + 0.25)
        L = rel[0] * vr[1] - rel[1] * vr[0]
        row.append(f"r{r:.2f} v{np.linalg.norm(vr):.2f} mu{mu:.1f} E{E:.2f} L{L:.1f}")
    print(" | ".join(row))
