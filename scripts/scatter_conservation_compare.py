"""Compare long-horizon conservation runs (scripts/scatter_conservation.py npz dirs) against a baseline dir, numpy only.

Usage: python scripts/scatter_conservation_compare.py --base results/conservation --cand results/cons_far_mono [--out results/cons_far_mono/compare.json]
Per regime: model median |dE|/(K0+|U0|), |dL|/Lscale, bound fraction, virial, outside-grid fraction, median radius, COM drift, max |dv|/dt at ticks 100/1000/1e4
for baseline model (b), candidate model (c) and truth (t, from the baseline npz)."""
import argparse
import glob
import json
import os

import numpy as np

CHECK = (100, 1000, 10000)


def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def at(arr, ticks, t):
    return arr[int(np.searchsorted(ticks, t))]


def stats(d, s, n, t):
    ticks = d["ticks"]
    K0, U0 = d[f"{s}_K"][0], d[f"{s}_U"][0]
    esc = K0 + np.abs(U0)
    lsc = d["r0"] * np.sqrt(2 * n * np.maximum(K0, 1e-12))
    g = lambda k: at(d[f"{s}_{k}"], ticks, t)
    return {"dE": med(np.abs(g("E") - K0 - U0) / esc), "dL": med(np.abs(g("L") - d[f"{s}_L"][0]) / lsc), "bound": med(g("bound")), "vir": med(g("vir")),
            "out": med(g("outgrid")), "rhalf": med(g("rhalf")), "rmax": med(g("rmax")),
            "com": med(np.hypot(g("comx") - d[f"{s}_comx"][0], g("comy") - d[f"{s}_comy"][0])), "nonfinite": int((d[f"{s}_first_nonfinite"] >= 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/conservation")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for f in sorted(glob.glob(f"{args.cand}/*.npz")):
        name = os.path.basename(f)[:-4]
        c = dict(np.load(f))
        b = dict(np.load(f"{args.base}/{name}.npz"))
        n = c["m_snaps"].shape[2]
        tmax = int(c["ticks"][-1])
        res[name] = {}
        print(f"== {name} (N={n}, B={len(c['seeds'])})")
        print(f"{'tick':>6s} {'quantity':>8s} {'base':>10s} {'cand':>10s} {'truth':>10s}")
        for t in [x for x in CHECK if x <= tmax]:
            sb, sc, st = stats(b, "m", n, t), stats(c, "m", n, t), stats(b, "t4", n, t)
            res[name][str(t)] = {"base": sb, "cand": sc, "truth": st}
            for k in ("dE", "dL", "bound", "vir", "out", "rhalf", "rmax", "com"):
                print(f"{t:6d} {k:>8s} {sb[k]:10.4g} {sc[k]:10.4g} {st[k]:10.4g}")
        print("  model nonfinite scenes base/cand:", stats(b, "m", n, tmax)["nonfinite"], stats(c, "m", n, tmax)["nonfinite"])
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
