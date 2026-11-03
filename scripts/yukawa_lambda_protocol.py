"""Row 1 follow-up: truth-blind Yukawa screening-length protocol (supervisor-log 2026-11-03 pre-registration). For yuk8, yuk32, yuk128 on each model, three
placement seeds, n placements per distance, the median radial-force curve is fitted with p and eps free and lambda fixed to each candidate in
{8, 32, 128, inf}; the candidate with the lowest rms log residual is the selected lambda. The free-lambda fit and the residual at the true lambda are stored.
Usage: PYTHONPATH=. python scripts/yukawa_lambda_protocol.py --n 1024 --out results/yukawa_lambda_protocol.json
"""
import argparse
import json

import numpy as np
import torch

from scripts.law_recovery_followup import DISTS_BASE, raw_curves
from scripts.law_recovery_probe import TRUTH, fit, load_multi

CANDS = (8.0, 32.0, 128.0, np.inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--models", default="mixed_s4738:ckpt/main_s4738.pt,mixed_s4739:ckpt/main_s4739.pt")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = {"n": a.n, "warm": a.warm, "candidates": [None if not np.isfinite(c) else c for c in CANDS], "models": {}}
    for spec in a.models.split(","):
        name, path = spec.split(":")
        model = load_multi(path, dev)
        res = out["models"][name] = {}
        for law in ("yuk8", "yuk32", "yuk128"):
            pt, et, lt = TRUTH[law]
            res[law] = {}
            for seed in (4738, 4739, 4740):
                y, _ = raw_curves(model, dev, law, DISTS_BASE, a.n, a.warm, seed)
                med = np.median(y, 1)
                fits = {("inf" if not np.isfinite(c) else str(int(c))): fit(DISTS_BASE, med, c) for c in CANDS}
                sel = min(fits, key=lambda k: fits[k]["rms_log_resid"])
                f = fits[sel]
                free = fit(DISTS_BASE, med, None)
                res[law][str(seed)] = {"selected": sel, "true": str(int(lt)), "correct": sel == str(int(lt)), "p_err": abs(f["p"] - pt) / pt, "eps_err": abs(f["eps"] - et) / et,
                                       "resid_by_candidate": {k: v["rms_log_resid"] for k, v in fits.items()},
                                       "free_fit": {"p": free["p"], "eps": free["eps"], "lam": free["lam"], "rms_log_resid": free["rms_log_resid"]}}
                print(name, law, seed, sel, "true", int(lt), "p_err %.3f eps_err %.3f" % (res[law][str(seed)]["p_err"], res[law][str(seed)]["eps_err"]),
                      {k: round(v, 3) for k, v in res[law][str(seed)]["resid_by_candidate"].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
