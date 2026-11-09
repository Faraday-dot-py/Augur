"""Verdict for the two-body-curve-scenes-finetune-vs-matched-control pre-registration (docs/debugging/experiment-log.md, 2026-11-09).
Pass: e_curve of pow1, pow3, lj each <= 0.5x control AND gravity err@20 <= 1.15x control. Kill: any e_curve > 0.8x control OR gravity err@20 > 1.30x control. Otherwise INCONCLUSIVE.
e_curve ratio = treatment / control from two_body_curve_scenes_finetune_eval.py; gravity err@20 = mean over the two scatter_multi_eval.py seeds of the scene-mean err20, treatment / control.
Usage: PYTHONPATH=. python scripts/score_two_body_curve_scenes_finetune.py --eval-dir results/two_body_curve_scenes_finetune --grav results/two_body_curve_scenes_finetune/gravity.json
"""
import argparse
import json
import math
import os
import sys

LAWS = ("pow1", "pow3", "lj")
E_PASS = 0.5
E_KILL = 0.8
G_PASS = 1.15
G_KILL = 1.30


def finite_pos(x, what):
    if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        sys.exit(f"BAD VALUE {what}: {x!r}")
    return float(x)


def grav_err20(res, arm):
    if arm not in res or "grav" not in res[arm]:
        sys.exit(f"MISSING gravity entry for {arm}")
    per = res[arm]["grav"]
    if not per:
        sys.exit(f"EMPTY gravity entry for {arm}")
    vals = []
    for seed, r in per.items():
        if not r.get("finite"):
            sys.exit(f"NON-FINITE rollout {arm} seed {seed}")
        vals.append(finite_pos(r["err20"], f"{arm} seed {seed} err20"))
    return sum(vals) / len(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--grav", required=True)
    a = ap.parse_args()
    curves = {}
    for arm in ("control", "treatment"):
        p = os.path.join(a.eval_dir, f"{arm}.json")
        if not os.path.exists(p):
            sys.exit(f"MISSING {p}")
        d = json.load(open(p))
        curves[arm] = {law: finite_pos(d[f"{arm}|{law}"]["e_curve"], f"{arm} {law} e_curve") for law in LAWS}
    if not os.path.exists(a.grav):
        sys.exit(f"MISSING {a.grav}")
    g = json.load(open(a.grav))
    gc, gc_seeds = grav_err20(g, "control")
    gt, gt_seeds = grav_err20(g, "treatment")
    e_ratio = {law: curves["treatment"][law] / curves["control"][law] for law in LAWS}
    g_ratio = gt / gc
    if any(r > E_KILL for r in e_ratio.values()) or g_ratio > G_KILL:
        verdict = "KILL"
    elif all(r <= E_PASS for r in e_ratio.values()) and g_ratio <= G_PASS:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    out = {"e_curve_ratio": e_ratio, "e_curve_control": curves["control"], "e_curve_treatment": curves["treatment"],
           "gravity_err20_control": gc, "gravity_err20_treatment": gt, "gravity_err20_ratio": g_ratio,
           "gravity_err20_control_per_seed": gc_seeds, "gravity_err20_treatment_per_seed": gt_seeds, "verdict": verdict}
    path = os.path.join(a.eval_dir, "score.json")
    json.dump(out, open(path, "w"), indent=1)
    if not os.path.getsize(path):
        sys.exit("EMPTY score.json")
    print(json.dumps(out, indent=1), flush=True)
    print("VERDICT", verdict, flush=True)


if __name__ == "__main__":
    main()
