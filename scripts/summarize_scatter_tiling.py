import json
import sys

import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else ""
force = json.load(open(f"results/scatter_tiling_force{tag}.json"))
roll = json.load(open(f"results/scatter_tiling_roll{tag}.json"))
CL = ("seam<2", "seam2-4", "seam4-8", "seam<8", "int8-16", "int>16", "interior", "outer<8")


def m(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


print("## Force (one step from shared single-grid states), held-out seeds, mean over scenes and ticks 0/5/10/20; rel err vs single grid")
print("columns: " + ", ".join(CL) + ", seam/interior (interior floored 1e-3), ghosts per tile")
for fam in ("cold300", "warm300", "warm1200"):
    rows = force[f"{fam}/held"]
    print(f"\n{fam}")
    names = [k for k in rows[0] if k not in ("seed", "rep", "tick", "single_vs_exact_by_g16_class")]
    for n in names:
        v = [m([r[n]["vs_single"][c]["rel"] for r in rows]) for c in CL]
        ratio = v[3] / max(v[6], 1e-3)
        print(f"  {n:16s} " + " ".join(f"{x:8.4f}" for x in v) + f"  ratio {ratio:6.2f}  ghosts {m([r[n]['ghosts'] for r in rows]):.0f}")
    se = [m([r["single_vs_exact_by_g16_class"][c]["rel"] for r in rows]) for c in ("seam<8", "interior")]
    print(f"  single-grid vs exact: seam<8 {se[0]:.3f} interior {se[1]:.3f}")
    for tk in (0, 20):
        rr = [r for r in rows if r["tick"] == tk]
        print(f"  tick {tk} seam<8/interior: " + " ".join(f"{n}={m([r[n]['vs_single']['seam<8']['rel'] for r in rr]):.4f}/{m([r[n]['vs_single']['interior']['rel'] for r in rr]):.4f}" for n in names if n in ("g16", "g16_fx", "g16_fx_far")))

print("\n## Rollouts (mean position error per body vs exact truth)")
for fam in ("cold300", "warm300", "warm1200"):
    print(f"\n{fam}: err@5 err@10 err@20 (ratio to single @20) | diff to single @20 | dP | seam/interior err20")
    for sp in ("held", "dev"):
        rows = roll[f"{fam}/{sp}"]
        names = [k for k in rows[0] if k not in ("seed", "rep", "constvel20")]
        base = m([r["single"]["err"]["20"] for r in rows])
        for n in names:
            e = [m([r[n]["err"][k] for r in rows]) for k in ("5", "10", "20")]
            print(f"  {sp:4s} {n:16s} {e[0]:8.4f} {e[1]:8.4f} {e[2]:8.4f} ({e[2] / base:5.2f}) diff {m([r[n]['diff_single']['20'] for r in rows]):8.4f} dP {m([r[n]['dP'] for r in rows]):.2e} "
                  f"s/i {m([r[n]['err20_by_class']['seam<8'] for r in rows]):.4f}/{m([r[n]['err20_by_class']['interior'] for r in rows]):.4f}")
        print(f"       constvel@20 {m([r['constvel20'] for r in rows]):.2f}")
