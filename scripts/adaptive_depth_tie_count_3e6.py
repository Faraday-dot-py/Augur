"""Counts, on the job 7475 scene (collapsed 3e6 Plummer cloud, seed 4738, fp32 coordinates), the bodies whose 16th and 17th nearest neighbour distances (r = sqrt(d^2 + 1e-8), fp64) are within a relative tol,
i.e. bodies where the top-16 set is ambiguous. Memory estimate: about 2 GB (cKDTree plus 3e6 x 19 distance and index arrays). CPU only.
Usage: PYTHONPATH=. python scripts/adaptive_depth_tie_count_3e6.py --out results/adaptive_depth_tie_and_single_query_diff_investigation/tie_count_3e6.json
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from scripts.pm_baseline_force_error import sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--core-scale", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(4738)
    p = sample("plummer", a.n, rng, a.core_scale).astype(np.float32).astype(np.float64)
    assert p.shape[0] == a.n
    dd, ii = cKDTree(p).query(p, k=19, workers=-1)
    r = np.sqrt(dd ** 2 + 1e-8)
    gap = (r[:, 17] - r[:, 16]) / r[:, 16]
    res = {"n": a.n, "body_1381234_gap_rel": float(gap[1381234])}
    for tol in (0.0, 1e-8, 1e-7, 1e-6, 1e-5):
        res[f"bodies_rank16_17_gap_rel_le_{tol}"] = int((gap <= tol).sum())
    res["expected_if_gap_density_uniform_over_median_rel_gap"] = float(np.median(gap))
    res["fraction_le_1e-6"] = float((gap <= 1e-6).mean())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
