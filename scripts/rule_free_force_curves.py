"""Rule-free force recovery, step 1: non-parametric radial force curves of the trained mixed-law scatter field for every law (docs/debugging/rule-free-force-recovery.md).
Per (checkpoint, law): median / mean / quartile radial acceleration of body 0 at each distance on the fit grid (three placement seeds) and on an interleaved held-out
grid (a fourth placement seed), plus the exact fp64 force at the same placements. One .npz per cell, skipped on rerun (resume).
Usage: PYTHONPATH=. python scripts/rule_free_force_curves.py --out results/rule_free_force_curves
"""
import argparse
import os

import numpy as np
import torch

from scripts.law_recovery_followup import raw_curves
from scripts.law_recovery_probe import load_multi

LAWS = ("pow1", "pow1.5", "pow2", "pow3", "yuk8", "yuk32", "yuk128", "lj")
D_FIT = np.geomspace(0.25, 14.0, 56)
D_HELD = np.sqrt(D_FIT[:-1] * D_FIT[1:])
FIT_SEEDS = (4738, 4739, 4740)
HELD_SEED = 4741


def summarise(y, yt):
    return {"med": np.median(y, 1), "mean": y.mean(1), "q25": np.percentile(y, 25, 1), "q75": np.percentile(y, 75, 1), "truth": np.median(yt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--models", default="mixed_s4738:ckpt/main_s4738.pt,mixed_s4739:ckpt/main_s4739.pt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    for spec in a.models.split(","):
        name, path = spec.split(":")
        model = load_multi(path, dev)
        for law in LAWS:
            f = os.path.join(a.out, f"{name}_{law}.npz")
            if os.path.exists(f):
                print("skip", f, flush=True)
                continue
            rec = {"d_fit": D_FIT, "d_held": D_HELD}
            for s in FIT_SEEDS:
                y, yt = raw_curves(model, dev, law, D_FIT, a.n, a.warm, s)
                for k, v in summarise(y, yt).items():
                    rec[f"fit_{s}_{k}"] = v
            y, yt = raw_curves(model, dev, law, D_HELD, a.n, a.warm, HELD_SEED)
            for k, v in summarise(y, yt).items():
                rec[f"held_{k}"] = v
            np.savez(f + ".tmp.npz", **rec)
            os.replace(f + ".tmp.npz", f)
            print(name, law, "fit med", np.round(rec["fit_4738_med"][::11], 4), "truth", np.round(rec["fit_4738_truth"][::11], 4), flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
