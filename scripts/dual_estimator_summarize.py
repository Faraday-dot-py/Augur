import json
import sys

import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/dual_estimator.json"))
print("fit", json.dumps(d["fit"]))


def at(rows, key, tgt, ykey="cost"):
    pts = sorted((r[key], r[ykey]) for r in rows)
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    if not (x.min() <= np.log(tgt) <= x.max()):
        return None
    return float(np.exp(np.interp(np.log(tgt), x, y)))


for s, r in d["eval"].items():
    print(f"== {s}")
    print("  geo   " + " ".join(f"th={x['theta']}:{x['rel_l2']:.4f}/p99 {x['p99']:.2f}/max {x['max']:.1f}/absp99 {x['abs_p99']:.3f}/absmax {x['abs_max']:.2f}/c{x['cost']:.0f}" for x in r["geo"]))
    for h in ("mean", "q90", "q99"):
        print(f"  {h:5s} " + " ".join(f"t={x['tol']:.0e}:{x['rel_l2']:.4f}/p99 {x['p99']:.2f}/max {x['max']:.1f}/absp99 {x['abs_p99']:.3f}/absmax {x['abs_max']:.2f}/c{x['cost']:.0f}" for x in r["est"] if x["head"] == h))
    for key in ("rel_l2", "abs_p99", "abs_max"):
        for tgt in ((0.01, 0.02, 0.04) if key == "rel_l2" else (0.03, 0.06, 0.1) if key == "abs_p99" else (0.1, 0.3, 1.0)):
            g = at(r["geo"], key, tgt)
            e = {h: at([x for x in r["est"] if x["head"] == h], key, tgt) for h in ("mean", "q90", "q99")}
            print(f"  cost @ {key}={tgt}: geo {g and round(g)} " + " ".join(f"{h} {v and round(v)}" for h, v in e.items()))
    for tol in sorted({x["tol"] for x in r["ctrl"]}):
        rows = [x for x in r["ctrl"] if x["tol"] == tol]
        last = rows[-4:]
        print(f"  ctrl tol={tol:.0e}: lam {' '.join(f'{x['lam']:.2f}' for x in rows)} | viol {' '.join(f'{x['violation']:.2f}' for x in rows)}")
        print(f"     last4 mean: rel_l2 {np.mean([x['rel_l2'] for x in last]):.4f} p99 {np.mean([x['p99'] for x in last]):.3f} max {np.mean([x['max'] for x in last]):.2f} abs_p99 {np.mean([x['abs_p99'] for x in last]):.3f} abs_max {np.mean([x['abs_max'] for x in last]):.2f} cost {np.mean([x['cost'] for x in last]):.0f}")
