"""Markdown tables for docs/debugging/substep-experiments.md from scripts/scatter_substep_exp.py output + conservation summaries.

  python scripts/scatter_substep_exp_summarize.py --acc results/substep_exp/acc.json --cons results/substep_exp/cons
"""
import argparse
import json
import os

import numpy as np

SEEDS = (9100, 9200, 9300)


def mean_err(acc, tag, i):
    return float(np.mean([acc[f"{tag}_{s}"]["err"][i] for s in SEEDS]))


def mean_e20(acc, tag, k):
    return float(np.mean([acc[f"{tag}_{s}"]["e20"][k] for s in SEEDS]))


def cons(d, key, reg):
    f = os.path.join(d, key.replace(":", "_"), "summary.json")
    if not os.path.exists(f):
        return float("nan"), float("nan")
    s = json.load(open(f))
    r = s.get(reg, {})
    m = r.get("m", {}).get("1000", {}).get("abs_dE_median", float("nan"))
    t = r.get("t4", {}).get("1000", {}).get("abs_dE_median", float("nan"))
    return m, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", required=True)
    ap.add_argument("--cons", required=True)
    args = ap.parse_args()
    R = json.load(open(args.acc))
    print("| model | scheme | ab @20 / @50 / @100 | cl @20 / @50 / @100 | cl @20 vs v1 | ab @20 vs v1 | e20 ab med / mean / p99 | e20 cl med / mean / p99 | nr100 dE@1000 | bh300 dE@1000 | force calls / sym pair passes per tick | ms/tick cl (B=24) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, acc in R["acc"].items():
        name, scheme = key.split(":")
        base = R["acc"].get(f"{name}:v1")
        ab = [mean_err(acc, "ab", i) for i in (2, 3, 4)]
        cl = [mean_err(acc, "cl", i) for i in (2, 3, 4)]
        g_cl = cl[0] / mean_err(base, "cl", 2) - 1 if base else float("nan")
        g_ab = ab[0] / mean_err(base, "ab", 2) - 1 if base else float("nan")
        ea = [mean_e20(acc, "ab", k) for k in ("median", "mean", "p99")]
        ec = [mean_e20(acc, "cl", k) for k in ("median", "mean", "p99")]
        nr, nrt = cons(args.cons, key, "nr100")
        bh, bht = cons(args.cons, key, "bh300")
        c = R["cost"][key]["cl"]
        fin = all(acc[k]["fin"] for k in acc)
        print(f"| {name} | {scheme} | {ab[0]:.4g} / {ab[1]:.3g} / {ab[2]:.3g} | {cl[0]:.4g} / {cl[1]:.3g} / {cl[2]:.3g} | {g_cl:+.0%} | {g_ab:+.0%} | "
              f"{ea[0]:.3g} / {ea[1]:.3g} / {ea[2]:.3g} | {ec[0]:.3g} / {ec[1]:.3g} / {ec[2]:.3g} | {nr:.3g} | {bh:.3g} | {c['force_per_tick']:.0f} / {c['pair_sym_per_tick']:.0f} | "
              f"{c['ms_per_tick']:.1f}{'' if fin else ' NONFINITE'} |")
    print()
    print("truth t4 dE@1000 floor (from any run): see cons summaries")
    if R.get("densities"):
        print("| state | within r<2 mean (p99) | knn16-capped | symmetric union (max) | within r<8 |")
        print("|---|---|---|---|---|")
        for k, v in R["densities"].items():
            print(f"| {k} | {v['within2']:.2f} ({v['within2_p99']:.0f}) | {v['knn16_capped']:.2f} | {v['sym_union']:.2f} ({v['sym_union_max']:.0f}) | {v['within8']:.1f} |")


if __name__ == "__main__":
    main()
