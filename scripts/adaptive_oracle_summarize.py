import json

import numpy as np

d = json.load(open("results/adaptive_oracle.json"))


def cost_at(rows, target, key="rel_l2"):
    pts = sorted((r[key], r["mean_cost"]) for r in rows)
    e = np.log([p[0] for p in pts])
    c = np.log([p[1] for p in pts])
    if not (e.min() <= np.log(target) <= e.max()):
        return None
    return float(np.exp(np.interp(np.log(target), e, c)))


for name, r in d.items():
    if name == "args":
        continue
    print(f"== {name} N={r['N']} extent={r['extent']:.0f} median_radius={r['median_radius']:.0f} exact_s={r['exact_time_s']:.1f} tree_build_s={r['tree_build_s']:.3f}")
    for m in r["mesh"]:
        print(f"  mesh g={m['grid']:5d} rel_l2={m['rel_l2']:.4f} p99={m['p99']:.3f} near/tgt={m['near_pairs_per_target']:.0f} t={m['near_s'] + m['far_s']:.4f}s net={m['net_ratio']:.1e}")
    for cap in (8, 32, 128):
        for m in [x for x in r["mac"] if x["cap"] == cap]:
            print(f"  mac cap={cap:3d} th={m['theta']:.2f} rel_l2={m['rel_l2']:.4f} p99={m['p99']:.3f} cost={m['mean_cost']:.0f} (mono {m['mean_mono']:.0f}) t={m['traverse_s']:.2f}s net={m['net_ratio']:.1e}")
    for m in r["oracle"]:
        print(f"  oracle cap={m['cap']:3d} eps={m['eps_node']:.0e} rel_l2={m['rel_l2']:.4f} p99={m['p99']:.3f} cost={m['mean_cost']:.0f} (mono {m['mean_mono']:.0f})")
    for tgt in (0.03, 0.01, 0.005):
        mac = {cap: cost_at([x for x in r["mac"] if x["cap"] == cap], tgt) for cap in (8, 32, 128)}
        ora = {cap: cost_at([x for x in r["oracle"] if x["cap"] == cap], tgt) for cap in (8, 32, 128)}
        print(f"  cost @ rel_l2={tgt}: mac {mac}  oracle {ora}")
