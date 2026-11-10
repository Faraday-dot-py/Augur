"""Which force component carries the net force on the black-hole core. Reads DIR/<stock>.json and DIR/<symmetrised>.json (stock and four-pass symmetrised arms, names given by --stock and --symmetrised) written by scripts/escape_roll.py with --probe-ticks.
Per probe tick t in TICKS and arm, on the class core15 (bodies within 15 units of the arm's own centre; raw force, momentum fix off): e = mean(a_model - a_exact), and the mean vectors of the
U-Net potential gradient u, the FFT kernel k and the pair term p. Switching a component off removes its mean: e_noU = e - u, e_noK = e - k, e_noP = e - p (reported).
Component X carries the bias at a tick when its mean projected on the error direction, (m_X . e) / |e|, is at least CARRY * |e|.
Classes (symmetrised arm, decided per tick, then majority over TICKS):
  stock check: median over TICKS of |e_stock| must be at least STOCK_X times |exact mean|; else INCONCLUSIVE.
  NO-NET-FORCE: |e_sym| < FLOOR * |e_stock| at that tick.   one carrier: UNET-CARRIES / KERNEL-CARRIES / PAIR-CARRIES.   two or three carriers: SHARED.   none: CANCELLATION.
  No class with at least MAJ of the TICKS: INCONCLUSIVE.
Usage: python scripts/core_force_component_verdict.py DIR --stock ARM --symmetrised ARM [--out DIR/core_force_component_verdict.json]"""
import argparse
import json
from collections import Counter

import numpy as np

TICKS = (100, 200, 300)
CLS = "core15"
CARRY = 0.5
FLOOR = 0.10
STOCK_X = 3.0
MIN_BODIES = 100
MAJ = 2
NAMES = {"u": "UNET-CARRIES", "k": "KERNEL-CARRIES", "p": "PAIR-CARRIES"}


def tick_vectors(doc, t):
    pr = next((p for p in doc["probes"] if p["t"] == t), None)
    if pr is None or pr["cls"].get(CLS) is None or pr["cls"][CLS]["n"] < MIN_BODIES or CLS not in pr.get("parts", {}):
        return None
    pa = pr["parts"][CLS]
    return {"e": np.array(pr["cls"][CLS]["bias"]), "u": np.array(pa["unet_mean"]), "k": np.array(pa["kernel_mean"]), "p": np.array(pa["pair_mean"]), "x": np.array(pa["exact_mean"]), "n": pr["cls"][CLS]["n"]}


def classify_tick(sym, stock):
    e = sym["e"]
    ne = float(np.linalg.norm(e))
    if ne < FLOOR * float(np.linalg.norm(stock["e"])):
        return "NO-NET-FORCE"
    carriers = [k for k in "ukp" if float(sym[k] @ e) / ne >= CARRY * ne]
    if not carriers:
        return "CANCELLATION"
    return NAMES[carriers[0]] if len(carriers) == 1 else "SHARED"


def tick_report(v):
    ne = float(np.linalg.norm(v["e"]))
    r = {"n_bodies": v["n"], "e": v["e"].tolist(), "e_norm": ne, "exact_mean_norm": float(np.linalg.norm(v["x"]))}
    for k in "ukp":
        r[f"mean_{k}"] = v[k].tolist()
        r[f"e_without_{k}_norm"] = float(np.linalg.norm(v["e"] - v[k]))
        r[f"proj_{k}_over_e"] = float(v[k] @ v["e"]) / ne / ne if ne > 0 else None
    return r


def verdict(stock_doc, sym_doc):
    per, classes, stock_ok = {}, [], []
    for t in TICKS:
        s, y = tick_vectors(stock_doc, t), tick_vectors(sym_doc, t)
        if s is None or y is None:
            per[str(t)] = {"skipped": "core15 under MIN_BODIES or probe missing"}
            continue
        c = classify_tick(y, s)
        classes.append(c)
        stock_ok.append(float(np.linalg.norm(s["e"])) / max(float(np.linalg.norm(s["x"])), 1e-12))
        per[str(t)] = {"class": c, "stock": tick_report(s), "sym": tick_report(y)}
    out = {"per_tick": per, "stock_over_exact_median": float(np.median(stock_ok)) if stock_ok else None}
    if not stock_ok or out["stock_over_exact_median"] < STOCK_X:
        out["verdict"] = "INCONCLUSIVE-stock-bias-not-reproduced"
        return out
    top, n = Counter(classes).most_common(1)[0]
    out["verdict"] = top if n >= MAJ else "INCONCLUSIVE-mixed-over-ticks"
    out["classes"] = classes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--stock", required=True)
    ap.add_argument("--symmetrised", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = verdict(json.load(open(f"{args.dir}/{args.stock}.json")), json.load(open(f"{args.dir}/{args.symmetrised}.json")))
    json.dump(res, open(args.out or f"{args.dir}/core_force_component_verdict.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in res if k != "per_tick"}))
    for t, r in res["per_tick"].items():
        if "class" in r:
            print(t, r["class"], f"|e_sym| {r['sym']['e_norm']:.4g} |e_stock| {r['stock']['e_norm']:.4g} no-U/K/P " + " ".join(f"{r['sym'][f'e_without_{k}_norm']:.4g}" for k in "ukp"))


if __name__ == "__main__":
    main()
