import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/granularity_estimator.json"))
b = json.load(open("results/adaptive_oracle.json"))
print("fit:", {s: (round(v["r2_mean"], 3), round(v["geo_only_r2"], 3), round(v["q_coverage"], 3)) for s, v in d["fit"].items()})
for s, r in d["eval"].items():
    print(f"== {s} ({r['split']})")
    for h in ("mean", "q90"):
        row = " ".join(f"tol={e['tol']:.0e}:{e['rel_l2']:.4f}/p99 {e['p99']:.3f}/max {e['max']:.2f}/c{e['mean_cost']:.0f}" for e in r["est"] if e["head"] == h and e["cap"] == 8)
        print(f"  {h:4s} cap8 rel_l2/p99/max/cost  {row}")
    print("  MAC cap8", " ".join(f"th={m['theta']}:{m['rel_l2']:.4f}/p99 {m['p99']:.3f}/max {m['max']:.2f}/c{m['mean_cost']:.0f}" for m in b[s]["mac"] if m["cap"] == 8))
    print("  oracle cap8", " ".join(f"eps={m['eps_node']:.0e}:{m['rel_l2']:.4f}/p99 {m['p99']:.3f}/max {m['max']:.2f}/c{m['mean_cost']:.0f}" for m in b[s]["oracle"] if m["cap"] == 8))
    for tgt, row in r["matched"].items():
        m, o, e = row["mac_cap8"], row["oracle_cap8"], row["est_mean_cap8"]
        print(f"  matched rel_l2={tgt}: mac {m and round(m)} oracle {o and round(o)} est {e and round(e)}  est/mac {e / m if m and e else None}")
