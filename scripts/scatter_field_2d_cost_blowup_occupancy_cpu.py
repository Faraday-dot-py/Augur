"""CPU-only count of pair-cell occupancy (cell edge 2.0) for the bench scenes, 2D (sigma 8, extent 64, clipped) and 3D Plummer (a 8), fixed box and constant density.
Usage: PYTHONPATH=. python scripts/scatter_field_2d_cost_blowup_occupancy_cpu.py <out.json>
"""
import json
import sys

import numpy as np

from scripts import scatter_scaling as ss
from scripts import three_dimensional_sparse_voxel_scatter_field as v3
from scripts.scatter_field_2d_cost_blowup_bench import SEED, box, box3, occupancy

res = {}
for n in (1000, 3000, 10000, 30000, 100000, 300000, 1000000):
    for scene in ("fixed-box", "constant-density-box"):
        sigma, extent = box(n, scene)
        p2, _ = ss.make_scene(n, sigma, 0.3, extent, SEED, 0)
        p3, _ = v3.plummer(n, box3(n, scene), 0.3, np.random.default_rng([SEED, n, 0]))
        res[f"{scene}|{n}"] = {"2d": occupancy(p2, 2.0, 2, extent, n), "3d": occupancy(p3, 2.0, 3, extent, n)}
        print(scene, n, res[f"{scene}|{n}"], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
