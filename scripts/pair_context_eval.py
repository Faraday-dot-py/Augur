"""Score neighbour-context-pair-branch-replacing-central-pair-term: frozen Exp E, matched control fine-tune and candidate fine-tune per training seed,
held-out Exp E scenes. Pre-registration: docs/debugging/architecture-push-prereg/neighbour-context-pair-branch-replacing-central-pair-term.md.
Usage: PYTHONPATH=. python scripts/pair_context_eval.py --e-ckpt E.pt --ckpt-dir ckpt --train-dir results/train --out results/pair_context/results.json
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_ft_data as fd
from scripts import train_scatter_field as tsf

STOCK = "ms_kp_pot_v_g128"
CAND = "ms_kp_pot_v_g128_pairctx"
STEPS = 30
ERR_PASS, ERR_KILL, TAN_PASS = 0.90, 0.95, 0.05
PLATEAU = 0.03
MOM_TICKS = (0, 10, 20)


def load(path, variant, dev):
    model = tsf.build(tsf.EXPS["E"], variant, 0.1).to(dev)
    sf.load_compat(model, torch.load(path, map_location=dev, weights_only=False)["model"])
    return model.eval()


@torch.no_grad()
def force_probe(model, data, dev):
    """Momentum error and tangential-to-radial pair output on truth states, before the momentum fix: per scene |sum m a| / sum m |a|, mean over scenes and ticks."""
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, dev)]
    model.momfix = False
    model.pp_stats = {} if model.pairctx else None
    errs = []
    for t in MOM_TICKS:
        a, _ = model.force(P[:, t], V[:, t], M, mask, model.init_field(P.shape[0], dev)[0])
        mm = (M * mask)[..., None]
        errs.append(float(((mm * a).sum(1).norm(dim=-1) / (mm * a.norm(dim=-1, keepdim=True)).sum((1, 2)).clamp_min(1e-12)).mean()))
    model.momfix = True
    tan = None
    if model.pairctx:
        tan = float(np.sqrt(model.pp_stats["tan_sq"] / model.pp_stats["rad_sq"]))
        model.pp_stats = None
    return float(np.mean(errs)), tan


def plateau(vals, iters):
    last = [v for i, v in vals if i > 0.9 * iters]
    prev = [v for i, v in vals if 0.8 * iters < i <= 0.9 * iters]
    if not last or not prev:
        raise SystemExit(f"validation log does not cover the last two 10% windows ({len(vals)} points)")
    decline = (np.mean(prev) - np.mean(last)) / np.mean(prev)
    return {"decline": float(decline), "undertrained": bool(decline > PLATEAU), "n_last": len(last), "n_prev": len(prev)}


def score(res, seeds, iters):
    ratios, moms, tans, plats = {}, {}, {}, {}
    for s in seeds:
        c, d = res["arms"][f"ctrl-s{s}"], res["arms"][f"cand-s{s}"]
        ratios[s] = d["err20_mean"] / c["err20_mean"]
        moms[s] = d["momentum_error_mean"] / c["momentum_error_mean"]
        tans[s] = d["tangential_over_radial"]
        plats[s] = {"cand": plateau(d["validation"], iters), "ctrl": plateau(c["validation"], iters)}
    pooled = float(np.mean(list(ratios.values())))
    mom_pooled = float(np.mean(list(moms.values())))
    kill = pooled >= ERR_KILL or mom_pooled > 1.0
    passed = all(r <= ERR_PASS for r in ratios.values()) and all(t < TAN_PASS for t in tans.values()) and not kill
    label = "PASS" if passed else "KILL" if kill else "INCONCLUSIVE"
    if label != "PASS" and any(p["cand"]["undertrained"] for p in plats.values()):
        label = "INCONCLUSIVE-undertrained"
    return {"label": label, "err20_ratio_per_train_seed": ratios, "err20_ratio_pooled": pooled, "momentum_ratio_per_train_seed": moms,
            "momentum_ratio_pooled": mom_pooled, "tangential_over_radial": tans, "plateau": plats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-ckpt", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--train-seeds", default="4738,4739")
    ap.add_argument("--eval-seeds", default="9100,9200,9300")
    ap.add_argument("--scenes", type=int, default=32)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    seeds = [int(s) for s in args.train_seeds.split(",")]
    eseeds = [int(s) for s in args.eval_seeds.split(",")]
    data = {e: fd.base(args.scenes, STEPS, e, dev) for e in eseeds}
    arms = {"frozen-exp-e": (args.e_ckpt, STOCK, None)}
    for s in seeds:
        arms[f"ctrl-s{s}"] = (f"{args.ckpt_dir}/pairctx-ctrl-s{s}.pt", STOCK, f"{args.train_dir}/pairctx-ctrl-s{s}_train.json")
        arms[f"cand-s{s}"] = (f"{args.ckpt_dir}/pairctx-cand-s{s}.pt", CAND, f"{args.train_dir}/pairctx-cand-s{s}_train.json")
    res = {"eval_seeds": eseeds, "train_seeds": seeds, "scenes": args.scenes, "ticks": STEPS, "iters": args.iters, "arms": {}}
    for name, (ck, var, tj) in arms.items():
        for p in (ck, tj):
            if p and not os.path.exists(p):
                raise SystemExit(f"missing {p}")
        model = load(ck, var, dev)
        per = {}
        for e in eseeds:
            m, _ = tsf.evaluate_model(model, data[e], dev, 0.1, 0.5)
            mom, tan = force_probe(model, data[e], dev)
            per[str(e)] = {"err5": m["err"][4], "err20": m["err"][19], "momentum_error": mom, "tangential_over_radial": tan,
                           "energy_drift_rel_30": m["energy_drift_rel_model"][30]}
        a = {"per_eval_seed": per, "err5_mean": float(np.mean([v["err5"] for v in per.values()])),
             "err20_mean": float(np.mean([v["err20"] for v in per.values()])),
             "momentum_error_mean": float(np.mean([v["momentum_error"] for v in per.values()]))}
        if var == CAND:
            a["tangential_over_radial"] = float(np.sqrt(np.mean([v["tangential_over_radial"] ** 2 for v in per.values()])))
        if tj:
            tr = json.load(open(tj))
            assert tr["iters"] == args.iters, f"{tj} ran {tr['iters']} iterations, expected {args.iters}"
            a["validation"] = tr["validation"]
        res["arms"][name] = a
        print(name, {k: v for k, v in a.items() if k not in ("per_eval_seed", "validation")}, flush=True)
    c20 = [res["arms"][f"ctrl-s{s}"]["per_eval_seed"][str(e)]["err20"] for s in seeds for e in eseeds]
    res["control_err20_spread"] = {"cv_all_cells": float(np.std(c20) / np.mean(c20)),
                                   "cv_across_train_seeds": float(np.std([res["arms"][f"ctrl-s{s}"]["err20_mean"] for s in seeds]) / np.mean([res["arms"][f"ctrl-s{s}"]["err20_mean"] for s in seeds])),
                                   "cv_across_eval_seeds": float(np.mean([np.std([res["arms"][f"ctrl-s{s}"]["per_eval_seed"][str(e)]["err20"] for e in eseeds]) / np.mean([res["arms"][f"ctrl-s{s}"]["per_eval_seed"][str(e)]["err20"] for e in eseeds]) for s in seeds]))}
    res["frozen_ratio_cand_over_frozen"] = {str(s): res["arms"][f"cand-s{s}"]["err20_mean"] / res["arms"]["frozen-exp-e"]["err20_mean"] for s in seeds}
    res["verdict"] = score(res, seeds, args.iters)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out + ".tmp", "w"), indent=1)
    os.replace(args.out + ".tmp", args.out)
    print("verdict", json.dumps(res["verdict"]), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
