"""Summarise results/scatter_balance.json (run 2 format): per scene/tick/scheme imbalance, work, makespan, batched ms, errors on fixed body sets.
Usage: python scripts/summarize_scatter_balance.py results/scatter_balance.json [--mean]   (--mean averages over scenes per tick/scheme)"""
import json
import sys

res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/scatter_balance.json"))
mean = "--mean" in sys.argv
cols = ("tiles", "own_max_over_mean", "tile_max_over_mean", "worker_max_over_mean", "makespan_ms", "total_work_ms", "batched_ms", "max_force_diff_vs_uniform")
rows = {}
for key, v in res.items():
    for tick, schemes in v["states"].items():
        for name, r in schemes.items():
            ref = r.get("force_vs_single_ref_classes", {})
            vals = [r.get(c) for c in cols] + [ref.get("all_in", {}).get("rel"), ref.get("useam<8", {}).get("rel"), ref.get("uint", {}).get("rel")]
            rows.setdefault((key.split("_")[0] if mean else key, tick, name), []).append(vals)
print("scene tick scheme | " + " | ".join(cols) + " | err_all | err_uniform_seam<8 | err_uniform_interior")
for k, lst in rows.items():
    out = []
    for i in range(len(lst[0])):
        xs = [x[i] for x in lst if x[i] is not None]
        out.append(sum(xs) / len(xs) if xs else None)
    print(" ".join(str(x) for x in k), "|", " | ".join("-" if x is None else (f"{x:.2e}" if abs(x) < 0.01 and x != 0 else f"{x:.2f}") for x in out))
for key, v in res.items():
    if "roll20" in v:
        print(key, "roll20", {k: round(x, 4) for k, x in v["roll20"].items()})
