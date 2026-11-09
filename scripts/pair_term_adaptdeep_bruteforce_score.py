"""Scores pair-term-adaptdeep-bruteforce-at-3e6 against the pre-registered line: PASS if adaptdeep error on the argmax body <= 1.7e-6, KILL if >= 1e-4, else INCONCLUSIVE.
Default is tie-aware: a body whose neighbour sets differ only through a 16th/17th distance tie (relative gap <= 1e-6) is scored against the closer of the two tied sets. --index-based gives the original comparison against the fp64 brute-force set.
Usage: python -I scripts/pair_term_adaptdeep_bruteforce_score.py results/pair_term_adaptdeep_bruteforce_at_3e6 [--index-based]
"""
import json
import os
import sys

PASS_MAX = 1.7e-6
KILL_MIN = 1e-4
TIE_REL = 1e-6

d = sys.argv[1]
tie_aware = "--index-based" not in sys.argv[2:]
assert os.path.exists(f"{d}/DONE"), "missing DONE marker"
r = json.load(open(f"{d}/result.json"))
assert r["done"], "result.json not marked done"


def score(b):
    e = b["err_adaptdeep_vs_fp64_bruteforce"]
    tie = b["fp64_top16_set_differs_from_fp32_top16_set"] and b["gap_r17_minus_r16_fp64"] / b["r16_fp64"] <= TIE_REL
    if tie_aware and tie:
        e = min(e, b["err_adaptdeep_vs_fp32_set_force"])
    return e, tie


e, argmax_tie = score(r["argmax"])
verdict = "PASS" if e <= PASS_MAX else "KILL" if e >= KILL_MIN else "INCONCLUSIVE"
print("mode", "tie-aware" if tie_aware else "index-based", "n", r["n"], "argmax body", r["argmax_body"], "count diff > 1e-6", r["count_diff_gt_1e-6"])
for b in r["bodies"]:
    s, tie = score(b)
    print(f"  body {b['body']} diff {b['abs_diff_default_vs_adaptdeep']:.3e} err_default {b['err_default_vs_fp64_bruteforce']:.3e} err_adaptdeep {b['err_adaptdeep_vs_fp64_bruteforce']:.3e} scored {s:.3e} tie {tie} set_differs {b['fp64_top16_set_differs_from_fp32_top16_set']} gap {b['gap_r17_minus_r16_fp64']:.3e}")
print(f"argmax adaptdeep error {e:.3e} tie {argmax_tie} (pass <= {PASS_MAX:g}, kill >= {KILL_MIN:g}): {verdict}")
