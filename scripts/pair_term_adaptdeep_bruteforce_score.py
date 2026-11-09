"""Scores pair-term-adaptdeep-bruteforce-at-3e6 against the pre-registered line: PASS if adaptdeep error on the argmax body <= 1.7e-6, KILL if >= 1e-4, else INCONCLUSIVE.
Usage: python -I scripts/pair_term_adaptdeep_bruteforce_score.py results/pair_term_adaptdeep_bruteforce_at_3e6
"""
import json
import os
import sys

PASS_MAX = 1.7e-6
KILL_MIN = 1e-4

d = sys.argv[1]
assert os.path.exists(f"{d}/DONE"), "missing DONE marker"
r = json.load(open(f"{d}/result.json"))
assert r["done"], "result.json not marked done"
e = r["argmax"]["err_adaptdeep_vs_fp64_bruteforce"]
verdict = "PASS" if e <= PASS_MAX else "KILL" if e >= KILL_MIN else "INCONCLUSIVE"
print("n", r["n"], "argmax body", r["argmax_body"], "count diff > 1e-6", r["count_diff_gt_1e-6"])
for b in r["bodies"]:
    print(f"  body {b['body']} diff {b['abs_diff_default_vs_adaptdeep']:.3e} err_default {b['err_default_vs_fp64_bruteforce']:.3e} err_adaptdeep {b['err_adaptdeep_vs_fp64_bruteforce']:.3e} set_differs {b['fp64_top16_set_differs_from_fp32_top16_set']} gap {b['gap_r17_minus_r16_fp64']:.3e}")
print(f"argmax adaptdeep error {e:.3e} (pass <= {PASS_MAX:g}, kill >= {KILL_MIN:g}): {verdict}")
