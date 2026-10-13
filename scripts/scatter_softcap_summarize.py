"""Pool results/softcap/*.json from scatter_softcap_eval.py: select seeds 9400+9500, held-out 9600; gravity runaway by channel.

  python scripts/scatter_softcap_summarize.py results/softcap/C.json results/softcap/D5ctrl.json
"""
import json
import sys

import numpy as np


def pool(rows):
    tm = np.array([max(r["tot"]) for r in rows])
    return dict(n=len(rows), run=float((tm > 200).mean()), a_med=float(np.median(tm)), e20=float(np.mean([r["err20"] for r in rows])), e100m=float(np.mean([r["err100"] for r in rows])),
                e100med=float(np.median([r["err100"] for r in rows])), off=float(np.mean([r["off100"] for r in rows])))


def fmt(p):
    return f"n{p['n']:3d} run {p['run']:.2f} |a|med {p['a_med']:8.1f} e20 {p['e20']:6.2f} e100 mean {p['e100m']:8.1f} med {p['e100med']:7.1f} off100 {p['off']:.2f}"


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        print("=====", path)
        for cap, r in d["caps"].items():
            for split, seeds in (("sel9400+9500", ("9400", "9500")), ("heldout9600", ("9600",))):
                con = [x for s in seeds for x in r["contact"].get(s, [])]
                if con:
                    print(f"s={cap:6s} {split:12s} contact  {fmt(pool(con))}")
                for fam in ("expB", "cluster"):
                    g = [x for s in seeds for x in r["gravity"].get(f"{fam}_{s}", [])]
                    if g:
                        print(f"s={cap:6s} {split:12s} {fam:8s} {fmt(pool(g))}")
            for fam in ("expB", "cluster"):
                g = [x for k, v in r["gravity"].items() if k.startswith(fam) for x in v]
                run = [x for x in g if max(x["tot"]) > 200]
                if not g:
                    continue
                ta = np.array([x["truth_a_max"] for x in g])
                print(f"s={cap:6s} gravity {fam}: truth max|a| median {np.median(ta):.1f} max {ta.max():.1f}; truth>200 in {(ta > 200).mean():.2f} of scenes")
                if run:
                    w = [x["worst"] for x in run]
                    legit = np.mean([x["truth_a_max"] > 200 for x in run])
                    print(f"      model runaway scenes {len(run)}/{len(g)}: truth also >200 in {legit:.2f}; worst body nn<1 {np.mean([b['worst_nn'] < 1 for b in w]):.2f}, off-grid(r>=32) "
                          f"{np.mean([b['worst_r'] >= 32 for b in w]):.2f}; |a| max median {np.median([max(x['tot']) for x in run]):.0f} (truth {np.median([x['truth_a_max'] for x in run]):.0f}); "
                          f"n bodies>200 median {np.median([b['n_gt200'] for b in w]):.0f} (off-grid {np.median([b['n_gt200_off'] for b in w]):.0f}, nn<1 {np.median([b['n_gt200_nn1'] for b in w]):.0f}); "
                          f"onset tick median {np.median([x['onset'] for x in run]):.0f}; field raw max median {np.median([max(x['fld']) for x in run]):.0f}; frac_off at worst median {np.median([b['frac_off'] for b in w]):.2f}")


main()
