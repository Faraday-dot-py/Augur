"""Print the seed-averaged tables of tier0_close_pair_localise.py output. Usage: python3 scripts/summarise_close_pair_localise.py <json>"""
import json
import sys

import numpy as np

res = json.load(open(sys.argv[1]))
seeds = [s for s in res]
methods = ["oracle_force_dt0.1_verlet", "baseline_tiled", "matched60k_halo", "new_halo"]


def avg(f):
    return np.mean([f(res[s]) for s in seeds], 0)


for tick in ("20", "100"):
    print(f"\n== dPE/|E0| by separation bin (model-truth), tick {tick}; pair-count ratio in brackets")
    for m in methods:
        rows = [avg(lambda r, i=i: [r[m]["pe_bins"][tick][i]["dPE_over_E0"], r[m]["pe_bins"][tick][i]["pair_count_ratio"]]) for i in range(5)]
        print(f"{m:28s}", "  ".join(f"{x[0]:+.4f}({x[1]:.2f})" for x in rows), f"sum {sum(x[0] for x in rows):+.4f}")
print("bins:", [b["r"] for b in res[seeds[0]]["new_halo"]["pe_bins"]["20"]])
for tick in ("20", "100"):
    print(f"\n== density tercile dPE per particle sum /sum|E0|, tick {tick}")
    for m in methods:
        print(f"{m:28s}", [round(float(avg(lambda r, i=i: r[m]["density_bins"][tick][i]["dPE_per_particle_over_E0"])), 4) for i in range(3)])
print("\n== bound pairs (truth r<1 at tick 20): frac still bound truth/model, mean log r ratio")
for tick in ("40", "60", "100"):
    for m in methods:
        b = avg(lambda r: [r[m]["bound_pairs"][tick][k] for k in ("frac_still_bound_truth", "frac_still_bound_model", "mean_log_r_model_over_truth")])
        print(tick, f"{m:28s}", np.round(b, 3))
print("\n== teacher-forced force gain / rel err by nearest-neighbour distance")
for tick in ("10", "20", "40"):
    for m in methods[1:]:
        g = avg(lambda r: [[x["mean_force_gain"], x["mean_rel_force_err"]] for x in r[m]["teacher_force"][tick]])
        print(tick, f"{m:18s}", "  ".join(f"{a:.3f}/{b:.3f}" for a, b in g))
