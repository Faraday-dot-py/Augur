import json

import numpy as np

d = json.load(open("results/dual_tree.json"))
b = json.load(open("results/adaptive_oracle.json"))


def cost_at(pts, target):
    pts = sorted(pts)
    e, c = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    if not (e.min() <= np.log(target) <= e.max()):
        return None
    return float(np.exp(np.interp(np.log(target), e, c)))


for s, r in d.items():
    if s == "args":
        continue
    print(f"== {s} nodes={r['nodes']}")
    for cap in (8, 32, 128):
        print(f"  dual cap={cap:3d} " + " ".join(f"th={x['theta']}:{x['rel_l2']:.4f}/p99 {x['p99']:.3f}/max {x['max']:.1f}/c{x['cost_per_particle']:.0f}(m2l {x['m2l_pairs_per_particle']:.0f}) net {x['net_ratio']:.0e}" for x in r["dual"] if x["cap"] == cap))
    line = {}
    for tgt in (0.01, 0.02, 0.05):
        line[tgt] = {"dual8": cost_at([(x["rel_l2"], x["cost_per_particle"]) for x in r["dual"] if x["cap"] == 8], tgt),
                     "dual32": cost_at([(x["rel_l2"], x["cost_per_particle"]) for x in r["dual"] if x["cap"] == 32], tgt),
                     "bh8": cost_at([(x["rel_l2"], x["mean_cost"]) for x in b[s]["mac"] if x["cap"] == 8], tgt)}
    print("  matched cost", {k: {kk: (round(vv) if vv else None) for kk, vv in v.items()} for k, v in line.items()})
    print(f"  time dual cap8: " + " ".join(f"{x['time_s']:.2f}" for x in r["dual"] if x["cap"] == 8), " | bh cap8: " + " ".join(f"{x['traverse_s']:.2f}" for x in b[s]["mac"] if x["cap"] == 8))
