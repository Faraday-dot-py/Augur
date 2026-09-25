import json

import numpy as np

import sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/dual_kernels.json"))


def at(rows, key, tgt):
    pts = sorted((r[key], r["cost"]) for r in rows)
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    if not (x.min() <= np.log(tgt) <= x.max()):
        return None
    return round(float(np.exp(np.interp(np.log(tgt), x, y))))


for k, r in d["kernels"].items():
    print(f"##### kernel {k}")
    print("  fit", {n: {kk: round(v, 3) for kk, v in f.items()} for n, f in r["fit"].items()})
    for s, e in r["eval"].items():
        print(f"  == {s}")
        print("    geo  ", " ".join(f"th={x['theta']}:{x['rel_l2']:.4f}/absp99 {x['abs_p99']:.3f}/c{x['cost']:.0f}/net {x['net_ratio']:.0e}" for x in e["geo"]))
        for h in ("own", "transfer"):
            rows = [x for x in e["est"] if x["head"] == h]
            if rows:
                print(f"    {h:8s}", " ".join(f"t={x['tol']:.0e}:{x['rel_l2']:.4f}/absp99 {x['abs_p99']:.3f}/c{x['cost']:.0f}" for x in rows))
        for key, tgts in (("rel_l2", (0.01, 0.02, 0.04)), ("abs_p99", (0.03, 0.06))):
            for t in tgts:
                g = at(e["geo"], key, t)
                own = at([x for x in e["est"] if x["head"] == "own"], key, t)
                tr = at([x for x in e["est"] if x["head"] == "transfer"], key, t) if any(x["head"] == "transfer" for x in e["est"]) else "-"
                print(f"    cost@{key}={t}: geo {g} own {own} transfer {tr}")

for k, r in d["kernels"].items():
    for s, e in r["eval"].items():
        for hn, rows in e.get("ctrl", {}).items():
            last = rows[-5:]
            print(f"CTRL {k} {s} head={hn}: lam {rows[0]['lam']:.2f}->{rows[-1]['lam']:.2f} viol first {rows[0]['violation']:.2f} last5 {sum(x['violation'] for x in last) / 5:.3f} | last5 rel_l2 {sum(x['rel_l2'] for x in last) / 5:.4f} abs_p99 {sum(x['abs_p99'] for x in last) / 5:.3f} cost {sum(x['cost'] for x in last) / 5:.0f} | first-iter rel_l2 {rows[0]['rel_l2']:.4f} abs_p99 {rows[0]['abs_p99']:.3f} cost {rows[0]['cost']:.0f}")
