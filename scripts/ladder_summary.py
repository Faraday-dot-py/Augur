"""Summarise ladder arms from results/el_eval. Usage: python scripts/ladder_summary.py [--gate TAG] TAG...
Per tag: ab/cl err@20/@100 (mean over eval seeds 9100/9200/9300), nr100 dE@1000 median of 3 regress runs, bh300 dE/dL@100 and @1000 medians.
Gate (exit 0 = PASS): nr100 dE@1000 <= 0.05 and ab err@20 <= 0.0076."""
import json
import os
import sys

import numpy as np

D = "results/el_eval"
SEEDS = (9100, 9200, 9300)


def row(tag):
    ev = json.load(open(f"{D}/warm_sq_{tag}_0_evalonly.json"))["eval"]
    out = {}
    for fam in ("ab", "cl"):
        for i, k in ((2, 20), (4, 100)):
            out[f"{fam}20" if k == 20 else f"{fam}100"] = float(np.mean([ev[f"{fam}_{s}"]["err"][i] for s in SEEDS]))
    reg = [json.load(open(f"{D}/regress_{tag}_r{r}.json")) for r in (1, 2, 3) if os.path.exists(f"{D}/regress_{tag}_r{r}.json")]
    med = lambda f: float(np.median([f(r) for r in reg]))
    out["nr100_dE1000"] = med(lambda r: r["conservation"]["nr100"]["model"]["dE"])
    out["nr100_dE1000_all"] = [round(r["conservation"]["nr100"]["model"]["dE"], 4) for r in reg]
    out["nr100_dL1000"] = med(lambda r: r["conservation"]["nr100"]["model"]["dL"])
    out["bh300_dE100"] = med(lambda r: r["regimes"]["bh300"]["mean"]["dE_rel_model"])
    out["bh300_dL100"] = med(lambda r: r["regimes"]["bh300"]["mean"]["dL_rel_model"])
    out["bh300_dE1000"] = med(lambda r: r["conservation"]["bh300"]["model"]["dE"])
    out["bh300_dL1000"] = med(lambda r: r["conservation"]["bh300"]["model"]["dL"])
    out["expB20"] = med(lambda r: r["regimes"]["expB"]["mean"]["err"]["20"])
    out["pass"] = bool(out["nr100_dE1000"] <= 0.05 and out["ab20"] <= 0.0076)
    return out


def main():
    args = sys.argv[1:]
    gate = None
    if args and args[0] == "--gate":
        gate, args = args[1], args[2:]
    cols = ["ab20", "ab100", "cl20", "cl100", "expB20", "nr100_dE1000", "nr100_dL1000", "bh300_dE100", "bh300_dL100", "bh300_dE1000", "bh300_dL1000", "pass"]
    print("| tag | " + " | ".join(cols) + " | nr100 dE@1000 x3 |")
    print("|" + "---|" * (len(cols) + 2))
    res = {}
    for t in args + ([gate] if gate and gate not in args else []):
        r = res[t] = row(t)
        print(f"| {t} | " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + f" | {r['nr100_dE1000_all']} |")
    if gate:
        sys.exit(0 if res[gate]["pass"] else 1)


if __name__ == "__main__":
    main()
