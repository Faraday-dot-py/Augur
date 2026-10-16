"""Summarize results/assim/assim.json: window errors per arm and family, ratio vs free."""
import json
import sys

d = json.load(open(sys.argv[1]))
for fam in ("solar", "star", "nr100"):
    if fam not in d:
        continue
    arms = d[fam]["arms"]
    free = arms["free"]["err_win"]
    print(f"== {fam} N={d[fam]['N'][:3]} T={d[fam]['T']}")
    for a, r in arms.items():
        w = r["err_win"]
        ratio = {k: (round(free[k] / w[k], 2) if w[k] and free[k] else None) for k in w if w[k] is not None}
        ws = {k: round(v, 4) for k, v in w.items() if v is not None}
        print(f"{a:12s} obs/body/tick {r['obs_per_body_tick']:.3f} err {ws} x-vs-free {ratio} dE {r['dE_vs_truth_last50']:.3g}")
    if fam == "solar":
        for a, r in arms.items():
            s = r["solar"]["planets"]
            print(f"  {a:12s} lost@100/300/1000 " + "/".join(f"{s[t]['lost_frac']:.2f}" for t in ("100", "300", "1000")) + " period_err@300/1000 " + "/".join(f"{s[t]['period_err_median']:.3f}" for t in ("300", "1000")))
