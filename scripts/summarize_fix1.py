"""Compact table from results/fix1_probe.json (offset 0 head-on rows)."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/fix1_probe.json"))
for off in (0.0, 0.6):
    print(f"offset {off}: model pass-through / relvy_ratio / relKE / pmom_err (truth in first row)")
    names = list(d)
    rows = {n: [r for r in d[n] if r["offset"] == off] for n in names}
    speeds = [r["speed_in"] for r in rows[names[0]]]
    t = rows[names[0]]
    print("speed  " + "  ".join(f"{s:>6}" for s in speeds))
    print("truth pt " + " ".join(f"{int(r['truth_passed_through']):>6}" for r in t))
    print("truth vy " + " ".join(f"{r['truth_relvy_ratio']:6.2f}" for r in t))
    for n in names:
        print(f"{n} pt   " + " ".join(f"{int(r['model_passed_through']):>6}" for r in rows[n]))
        print(f"{n} vy   " + " ".join(f"{r['model_relvy_ratio']:6.2f}" for r in rows[n]))
        print(f"{n} KE   " + " ".join(f"{r['model_relKE_ratio']:6.2f}" for r in rows[n]))
        print(f"{n} pmom " + " ".join(f"{r['model_pmom_err']:6.1f}" for r in rows[n]))
