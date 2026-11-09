"""Held-out-grid force-curve error (e_curve of scripts/score_rule_free_force_recovery.py:55) for pow1, pow3, lj on the fine-tune arm checkpoints.
Usage: PYTHONPATH=. python scripts/two_body_curve_scenes_finetune_eval.py --models base:ckpt/main_s4739.pt,control:...,treatment:... --out results/two_body_curve_scenes_finetune
"""
import argparse
import json
import os
import sys

import numpy as np

from scripts import rule_free_force_curves as rfc
from scripts.rule_free_force_discovery import exact_curve, zt

LAWS = ("pow1", "pow3", "lj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rfc.LAWS = LAWS
    curves = os.path.join(a.out, "curves")
    sys.argv = ["rule_free_force_curves.py", "--out", curves, "--models", a.models]
    rfc.main()
    res = {}
    for spec in a.models.split(","):
        name = spec.split(":")[0]
        for law in LAWS:
            z = np.load(os.path.join(curves, f"{name}_{law}.npz"))
            tf, th = exact_curve(law, rfc.D_FIT), exact_curve(law, rfc.D_HELD)
            cs = 1e-3 * np.max(np.abs(tf))
            e_curve = float(np.sqrt(np.mean((zt(z["held_med"], cs) - zt(th, cs)) ** 2)))
            S = float(np.sqrt(np.mean((zt(z["fit_4738_med"], cs) - zt(z["fit_4739_med"], cs)) ** 2)))
            res[f"{name}|{law}"] = {"e_curve": e_curve, "S": S, "ratio": e_curve / max(S, 0.01)}
            print(name, law, res[f"{name}|{law}"], flush=True)
    for name in sorted({k.split("|")[0] for k in res}):
        json.dump({k: v for k, v in res.items() if k.startswith(name + "|")}, open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
