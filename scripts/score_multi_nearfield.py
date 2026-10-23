"""Score the near-field retrain against the pre-registration (docs/debugging/experiment-log.md, 2026-10-22).
Usage: python scripts/score_multi_nearfield.py --main eval_main.json --cmp eval_cmp1.json --cmp eval_cmp2.json
"""
import argparse
import json

import numpy as np

SEEDS = ("9000", "12000")
PAIRS = {"HELD_pow1.5": "cmp_pow1.5_single", "HELD_yuk32": "cmp_yuk32_single", "HELD_charge_yuk32": "cmp_yuk32_charge",
         "HELD_pp_pow1.5": "cmp_pow1.5_predprey"}


def mean(d, case, key):
    return float(np.mean([d[case][s][key] for s in SEEDS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--cmp", action="append", required=True)
    a = ap.parse_args()
    main_d = json.load(open(a.main))
    cmp_d = {}
    for f in a.cmp:
        cmp_d.update(json.load(open(f)))
    e = main_d["E"]
    print("gravity (64-extent single grid): force_rel / err20 ; E", round(mean(e, "grav", "force_rel"), 3), round(mean(e, "grav", "err20"), 4))
    for seed in ("s4738", "s4739"):
        m = main_d[seed]
        gr = mean(m, "grav", "err20") / mean(e, "grav", "err20")
        verdict_grav = "KILL(>1.25x)" if gr > 1.25 else "ok"
        print(f"\n{seed}: grav force_rel {mean(m, 'grav', 'force_rel'):.3f} err20 {mean(m, 'grav', 'err20'):.4f} ratio-vs-E {gr:.2f}x {verdict_grav}")
        for case, cmp_name in PAIRS.items():
            if cmp_name not in cmp_d:
                print(f"  {case}: comparator {cmp_name} missing")
                continue
            c = cmp_d[cmp_name]
            fr = mean(m, case, "force_rel") / mean(c, case, "force_rel")
            er = mean(m, case, "err20") / mean(c, case, "err20")
            print(f"  {case}: force_rel main {mean(m, case, 'force_rel'):.3f} cmp {mean(c, case, 'force_rel'):.3f} ratio {fr:.2f} "
                  f"({'PASS<=1.5' if fr <= 1.5 else 'KILL>3' if fr > 3 else 'between'}); err20 main {mean(m, case, 'err20'):.3f} cmp "
                  f"{mean(c, case, 'err20'):.3f} ratio {er:.2f} ({'PASS<=2' if er <= 2 else 'fail'})")


if __name__ == "__main__":
    main()
