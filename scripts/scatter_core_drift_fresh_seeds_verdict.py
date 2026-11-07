"""Verdict for scatter-core-drift-arm-a-fresh-held-out-seeds against the original arm A gates (experiment-log 2026-10-22), per fresh seed.
Gates: core drift <= .007/tick (N=2e4), err@20 <= 1.05x stock E (two_body, expB, bh300), bh300 dE@1000 ratio <= 1.15x stock E, nr100 judged by the lower end of its 95% bootstrap interval (noise 1.37x in the original).
KILL if any value exceeds 1.5x its gate; PASS if every gate is met on every valid seed and the pooled bh300 dE ratio is <= 1.15/1.05 (stock-vs-stock noise 1.05x); else INCONCLUSIVE.
A seed is valid for drift only if the stock-E arm drifts >= .03/tick there (the bug is exercised); PASS needs at least 2 valid drift seeds.
Usage: python scripts/scatter_core_drift_fresh_seeds_verdict.py --cons-stock D --cons-sym D --reg-stock J --reg-sym J --drift-dirs D [D ...]"""
import argparse
import json
import re
import sys

import numpy as np

from scripts.summarize_cons_compare import dE_at, med

SEEDS = (12100, 12200, 12300)
GATE_DRIFT, GATE_ERR, GATE_DE, KILL_MULT, DE_NOISE, VALID_DRIFT = 0.007, 1.05, 1.15, 1.5, 1.05, 0.03
REGIMES = ("two_body", "expB", "bh300")


def drift_of(path, arm):
    for line in open(path):
        m = re.match(rf"{arm}\s+drift (-?[0-9.]+)/tick", line)
        if m:
            return float(m.group(1))
        if line.startswith(f"{arm} ") and "core lost" in line:
            return float("inf")
    sys.exit(f"no {arm} line in {path}")


def dE_ratios(stock, sym, regime):
    a = dict(np.load(f"{stock}/{regime}.npz"))
    b = dict(np.load(f"{sym}/{regime}.npz"))
    assert (a["seeds"] == b["seeds"]).all()
    da, db = dE_at(a, 1000), dE_at(b, 1000)
    if not (np.isfinite(da).all() and np.isfinite(db).all()):
        sys.exit(f"non-finite dE@1000 in {regime}: {int((~np.isfinite(da)).sum())} stock, {int((~np.isfinite(db)).sum())} sym scenes")
    rng = np.random.default_rng(4738)
    out = {}
    for s in SEEDS:
        m = a["seeds"] == s
        n = int(m.sum())
        idx = rng.integers(0, n, (2000, n))
        boot = np.median(db[m][idx], 1) / np.median(da[m][idx], 1)
        out[s] = (med(db[m]) / med(da[m]), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), n)
    out["pooled"] = (med(db) / med(da), None, None, len(da))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cons-stock", required=True)
    ap.add_argument("--cons-sym", required=True)
    ap.add_argument("--reg-stock", required=True)
    ap.add_argument("--reg-sym", required=True)
    ap.add_argument("--drift-dirs", nargs=len(SEEDS), required=True)
    args = ap.parse_args()
    rows = []
    rs, rc = json.load(open(args.reg_stock)), json.load(open(args.reg_sym))
    for reg in REGIMES:
        for s in SEEDS:
            a = rs["regimes"][reg]["per_seed"][str(s)]["err"]["20"]
            b = rc["regimes"][reg]["per_seed"][str(s)]["err"]["20"]
            rows.append((f"err@20 {reg} seed {s}", b / a, GATE_ERR))
    pooled = {}
    for reg in ("bh300", "nr100"):
        r = dE_ratios(args.cons_stock, args.cons_sym, reg)
        pooled[reg] = r["pooled"][0]
        for s in SEEDS:
            ratio, lo, hi, n = r[s]
            print(f"dE@1000 {reg} seed {s}: ratio {ratio:.3f} (95% boot {lo:.3f}..{hi:.3f}, {n} scenes)")
            rows.append((f"dE@1000 {reg} seed {s}" + (" (boot low)" if reg == "nr100" else ""), lo if reg == "nr100" else ratio, GATE_DE))
        print(f"dE@1000 {reg} pooled ratio {r['pooled'][0]:.3f} ({r['pooled'][3]} scenes)")
    valid = 0
    for s, d in zip(SEEDS, args.drift_dirs):
        base, fix = drift_of(f"{d}/drift.txt", "S1"), drift_of(f"{d}/drift.txt", "S1c4")
        print(f"drift seed {s}: stock E {base:.4f}/tick, symc4 {fix:.4f}/tick")
        if base < VALID_DRIFT:
            print(f"  seed {s} invalid for drift (stock E below {VALID_DRIFT}), excluded")
            continue
        valid += 1
        rows.append((f"drift seed {s}", fix, GATE_DRIFT))
    kill, over = [], []
    for name, v, g in rows:
        status = "KILL" if v > KILL_MULT * g else ("fail" if v > g else "ok")
        print(f"{name:32s} {v:9.4f} gate {g:<6g} kill line {KILL_MULT * g:<8.4g} {status}")
        if status == "KILL":
            kill.append(name)
        elif status == "fail":
            over.append(name)
    thin = pooled["bh300"] > GATE_DE / DE_NOISE
    if kill:
        verdict = f"KILL ({', '.join(kill)})"
    elif over or thin or valid < 2:
        why = over + (["pooled bh300 dE ratio in noise band"] if thin else []) + ([f"only {valid} valid drift seeds"] if valid < 2 else [])
        verdict = f"INCONCLUSIVE ({', '.join(why)})"
    else:
        verdict = "PASS"
    print("VERDICT", verdict)


if __name__ == "__main__":
    main()
