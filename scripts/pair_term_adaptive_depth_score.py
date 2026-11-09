"""Scores pair-term-adaptive-depth-fix (job 7235) against the pre-registered pass/kill line in docs/debugging/experiment-log.md.
Usage: python -I scripts/pair_term_adaptive_depth_score.py results/pair_term_adaptive_depth_fix
"""
import json
import sys

d = sys.argv[1]
REF_T = 0.397
for tag in ("collapsed-1e6", "reference-3e6", "collapsed-3e6"):
    r = json.load(open(f"{d}/{tag}.json"))
    print(tag, "n", r["n"], "core", r["core_scale"], "ckpt", r["ckpt"], "rms_acc", r["ref_rms_acc"], "bit_identical_to_head", r.get("default_bit_identical_to_head"))
    cur = r["variants"].get("default")
    for name, v in r["variants"].items():
        fb = v["fallback"]
        frac = None if fb is None else fb / r["n"]
        t = v["steady_mean_s"]
        floor = 1e-5 * r["ref_rms_acc"]
        err_ok = v["max_abs_err_vs_bruteforce"] <= cur["max_abs_err_vs_bruteforce"] + floor
        err_kill = v["max_abs_err_vs_bruteforce"] > 1.5 * cur["max_abs_err_vs_bruteforce"] + floor
        ratio = "n/a" if t is None else f"{t / REF_T:.2f}"
        print(f"  {name}: fallback {fb} frac {frac} steady_s {t} ({ratio}x) diff_vs_default {v['max_abs_diff_vs_default_all_bodies']:.3e} max_err {v['max_abs_err_vs_bruteforce']:.3e} cur {cur['max_abs_err_vs_bruteforce']:.3e} err_ok {err_ok} err_kill {err_kill} bit_equal {v['bit_equal_to_default']}")
