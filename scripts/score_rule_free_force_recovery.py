"""Rule-free force recovery, step 3: apply the lines pre-registered in docs/debugging/experiment-log.md (2026-11-08) to the saved curves and discovery results.
Usage: PYTHONPATH=. python scripts/score_rule_free_force_recovery.py --curves results/rule_free_force_curves --exact results/rule_free_force_exact.json --model results/rule_free_force_model.json
"""
import argparse
import json
import os

import numpy as np

from scripts.rule_free_force_discovery import BASES, D_FIT, D_HELD, LAWS, exact_curve, zt

TRUE_NAMES = {"pow": ("soft-power",), "yuk": ("soft-power-exponential",), "lj": ("two-power+sat", "two-power+clamp+sat")}
TRUTH_PARAMS = {"pow1": [1.0, 0.5], "pow1.5": [1.5, 0.5], "pow2": [2.0, 0.5], "pow3": [3.0, 0.5], "yuk8": [2.0, 0.5, 8.0], "yuk32": [2.0, 0.5, 32.0], "yuk128": [2.0, 0.5, 128.0], "lj": [7.0, 13.0]}
HELD_OUT = ("pow1.5", "yuk32")


def phys(c):
    out = []
    for (kind, _, _), v in zip(BASES[c["base"]], c["x"]):
        if kind in ("expo", "soft"):
            out.append(v)
        elif kind == "logL":
            out.append(10.0 ** v)
    return sorted(out) if c["base"] == "two-power" else out


def fam(law):
    return TRUE_NAMES["pow" if law.startswith("pow") else "yuk" if law.startswith("yuk") else "lj"]


def verdict(n_pass, n_kill, total=16):
    if n_pass >= 12 and n_kill <= 1:
        return "PASS"
    return "KILL" if n_kill >= total // 2 else "INCONCLUSIVE_PARTIAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    ex = json.load(open(a.exact))
    md = json.load(open(a.model))
    gate = sum(ex[f"exact|{l}"]["selected"]["E_held"] <= 0.05 for l in LAWS)
    print("validity gate: laws with E_x <= 0.05:", gate, "of 8", "OK" if gate >= 7 else "INVALID")
    l1 = {"pass": 0, "kill": 0}
    l2 = {"pass": 0, "kill": 0}
    cells = []
    for ck in ("mixed_s4738", "mixed_s4739"):
        for law in LAWS:
            z = np.load(os.path.join(a.curves, f"{ck}_{law}.npz"))
            tf, th = exact_curve(law, D_FIT), exact_curve(law, D_HELD)
            cs = 1e-3 * np.max(np.abs(tf))
            e_curve = float(np.sqrt(np.mean((zt(z["held_med"], cs) - zt(th, cs)) ** 2)))
            S = float(np.sqrt(np.mean((zt(z["fit_4738_med"], cs) - zt(z["fit_4739_med"], cs)) ** 2)))
            r1 = e_curve / max(S, 0.01)
            rel = (z["held_med"] - th) / np.abs(th)
            bands = [float(np.median(rel[(D_HELD >= lo) & (D_HELD < hi)])) for lo, hi in ((0, 1), (1, 4), (4, 15))]
            c1 = "PASS" if r1 <= 3 else "KILL" if r1 > 10 else "MARGINAL"
            l1["pass"] += c1 == "PASS"
            l1["kill"] += c1 == "KILL"
            sels = [md[f"{ck}|{law}|{s}"]["selected"] for s in (4738, 4739, 4740)]
            sel = sels[0]
            xs = ex[f"exact|{law}"]
            D = max(xs["selected"]["E_held"], 0.01)
            r2 = sel["E_held"] / D
            c2 = "PASS" if r2 <= 3 else "KILL" if r2 > 10 else "MARGINAL"
            l2["pass"] += c2 == "PASS"
            l2["kill"] += c2 == "KILL"
            same_exact = sel["name"] == xs["selected"]["name"]
            same_true = sel["name"] in fam(law)
            tp = TRUTH_PARAMS[law]
            within = same_true and len(phys(sel)) == len(tp) and all(abs(p - t) / t <= 0.2 for p, t in zip(phys(sel), tp))
            names = {s["name"] for s in sels}
            spread = None
            if len(names) == 1:
                P = np.array([phys(s) for s in sels])
                spread = (np.ptp(P, 0) / P.mean(0)).tolist() if P.shape[1] else []
            unident = len(names) > 1 or (spread is not None and any(v > 0.2 for v in spread)) or len(xs["tied"]) > 1
            cells.append({"ckpt": ck, "law": law, "held_out_from_training": law in HELD_OUT, "level1_e_curve": e_curve, "level1_S": S, "level1_ratio": r1, "level1": c1, "bias_by_band": bands,
                          "E_m": sel["E_held"], "E_x": xs["selected"]["E_held"], "R": r2, "level2": c2, "selected": sel["name"], "expr": sel["expr"], "exact_run": xs["selected"]["name"], "exact_expr": xs["selected"]["expr"],
                          "same_as_exact_run": same_exact, "same_family_as_true": same_true, "params_within_20pct_of_truth": within, "replicate_names": [s["name"] for s in sels],
                          "replicate_param_spread": spread, "replicate_E_held": [s["E_held"] for s in sels], "unidentifiable": bool(unident)})
            print(ck, law, "| L1 e_curve %.3f S %.3f ratio %.1f %s | L2 E_m %.3f R %.1f %s | %s | exact-run %s same %s true-family %s params20 %s unident %s" %
                  (e_curve, S, r1, c1, sel["E_held"], r2, c2, sel["name"], xs["selected"]["name"], same_exact, same_true, within, unident), flush=True)
            print("    ", sel["expr"], "| exact-run:", xs["selected"]["expr"], "| seeds", [s["name"] for s in sels], "| bias bands", np.round(bands, 3))
    print("LEVEL ONE pass", l1["pass"], "kill", l1["kill"], verdict(l1["pass"], l1["kill"]))
    print("LEVEL TWO pass", l2["pass"], "kill", l2["kill"], verdict(l2["pass"], l2["kill"]))
    json.dump({"gate": gate, "level1": l1, "level2": l2, "cells": cells}, open("results/rule_free_force_recovery_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
