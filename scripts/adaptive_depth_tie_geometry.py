"""Regenerates the job 7475 scene (collapsed 3e6 Plummer cloud, seed 4738) on CPU and prints the geometry of the tied 16th/17th neighbours of one body:
positions, displacement vectors, unit-vector difference, and the Newtonian 1/r^2 magnitude, to judge whether a 1.47e-2 force gap between the two tied sets is plausible.
Usage: PYTHONPATH=. python scripts/adaptive_depth_tie_geometry.py --out results/adaptive_depth_tie_and_single_query_diff_investigation/geometry.json
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts.pm_baseline_force_error import sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--core-scale", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--body", type=int, default=1381234)
    ap.add_argument("--tied", type=int, nargs=2, default=[627953, 849166])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pos = torch.tensor(sample("plummer", a.n, rng, a.core_scale), dtype=torch.float32)
    assert pos.shape[0] == a.n
    p64 = pos.double()
    b = a.body
    res = {"body": b, "pos_fp32": pos[b].tolist(), "pos_fp32_norm": float(pos[b].norm())}
    r64 = torch.sqrt(((p64 - p64[b]) ** 2).sum(-1) + 1e-8)
    r64[b] = 1e6
    order = torch.argsort(r64)
    res["fp64_top18"] = [(int(i), float(r64[i])) for i in order[:18]]
    d = {}
    for t in a.tied:
        ds = p64[t] - p64[b]
        d[t] = ds
        res[f"tied_{t}"] = {"pos_fp32": pos[t].tolist(), "ds_fp64": ds.tolist(), "ds_from_fp32_coords_exact": ds.tolist(), "r": float(r64[t]), "newton_w_1_over_r2": float(1 / r64[t] ** 2)}
    u0, u1 = d[a.tied[0]] / r64[a.tied[0]], d[a.tied[1]] / r64[a.tied[1]]
    diff = float((u0 - u1).norm())
    res["unit_vector_difference_norm"] = diff
    res["angle_between_rad"] = float(torch.acos((u0 * u1).sum().clamp(-1, 1)))
    res["distance_between_tied_bodies"] = float((p64[a.tied[0]] - p64[a.tied[1]]).norm())
    res["fp32_ulp_at_body_coords"] = [float(torch.nextafter(pos[b][i], torch.tensor(1e9)) - pos[b][i]) for i in range(2)]
    res["newton_force_swap_magnitude_if_w_is_1_over_r2_mass_1"] = diff / float(r64[a.tied[0]]) ** 2
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
