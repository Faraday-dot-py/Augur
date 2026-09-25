"""Audit-noise study from results/fallback_fa_*_adapt.json (`noise` records: full-N true error vs 100 random k-subsets audits).
Usage: python scripts/fallback_noise_summarize.py"""
import glob
import json

import numpy as np

tgt = 0.01
rows = {}
for f in sorted(glob.glob("results/fallback_fa_*_adapt.json")):
    o = json.load(open(f))
    for r in o["noise"]:
        rows.setdefault(r["k"], []).append({**r, "ic": o["cfg"]["ic"]})
print("k  n  true_over_target_frac  rel_std(audit)/true  misclass(true<=tgt: audit>tgt)  misclass(true>tgt: audit<=tgt)  near[0.7-1.4x]: misclass  p_severe(mean)")
for k in sorted(rows):
    r = rows[k]
    tr = np.array([x["true"] for x in r])
    po = np.array([x["p_over"] for x in r])
    ps = np.array([x["p_severe"] for x in r])
    rs = np.array([x["std"] / x["true"] for x in r])
    ok, bad = tr <= tgt, tr > tgt
    near = (tr > 0.7 * tgt) & (tr < 1.4 * tgt)
    mis = np.where(ok, po, 1 - po)
    print(k, len(r), f"{bad.mean():.3f}", f"{rs.mean():.3f}", f"{po[ok].mean() if ok.any() else float('nan'):.3f}", f"{(1 - po[bad]).mean() if bad.any() else float('nan'):.3f}",
          f"{mis[near].mean() if near.any() else float('nan'):.3f} (n={near.sum()})", f"{ps.mean():.4f}")
for ic in sorted({x['ic'] for v in rows.values() for x in v}):
    for k in sorted(rows):
        r = [x for x in rows[k] if x["ic"] == ic]
        print(ic, k, "true mean/max", f"{np.mean([x['true'] for x in r]):.4f}/{np.max([x['true'] for x in r]):.4f}", "rel std", f"{np.mean([x['std'] / x['true'] for x in r]):.3f}", "p_over mean", f"{np.mean([x['p_over'] for x in r]):.3f}")
