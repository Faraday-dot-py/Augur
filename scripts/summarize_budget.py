"""Print a comparison table from downloaded budget-run jsons. Usage: python scripts/summarize_budget.py name=path ..."""
import json
import os
import sys

print("| run | iters | s | 9000 @5/10/20 | 12000 @5/10/20 | @50 | @100 | dE/E@100 | dL/L@100 | dP@100 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in sys.argv[1:]:
    name, path = a.split("=")
    r = json.load(open(os.path.expanduser(path)))
    m, m2 = r["9000"], r["12000"]
    f = lambda x: "/".join(f"{x['err'][i]:.4f}" for i in (4, 9, 19))
    it = r.get("train_iters", "-")
    sec = r.get("train_seconds", "-")
    sec = f"{sec:.0f}" if sec != "-" else sec
    print(f"| {name} | {it} | {sec} | {f(m)} | {f(m2)} | {m['err'][49]:.3f} | {m['err'][99]:.3f} | {m['energy_drift_rel_model'][99]:.3f} | {m['angmom_drift_rel_model'][99]:.3f} | {m['mom_drift_model'][99]:.1e} |")
