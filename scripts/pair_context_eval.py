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


def score_lines(res, seeds, iters, lines):
    """seed-replication: PASS err20 ratio <= 0.90 on every train seed, KILL ratio > 0.95 on any. tangential-penalty: PASS tangential/radial < 0.05 and ratio <= 0.90, KILL ratio > 0.95."""
    ratios = {s: res["arms"][f"cand-s{s}"]["err20_mean"] / res["arms"][f"ctrl-s{s}"]["err20_mean"] for s in seeds}
    tans = {s: res["arms"][f"cand-s{s}"]["tangential_over_radial"] for s in seeds}
    moms = {s: res["arms"][f"cand-s{s}"]["momentum_error_mean"] / res["arms"][f"ctrl-s{s}"]["momentum_error_mean"] for s in seeds}
    plats = {s: {"cand": plateau(res["arms"][f"cand-s{s}"]["validation"], iters), "ctrl": plateau(res["arms"][f"ctrl-s{s}"]["validation"], iters)} for s in seeds}
    kill = any(r > ERR_KILL for r in ratios.values())
    ok = all(r <= ERR_PASS for r in ratios.values())
    if lines == "tangential-penalty":
        ok = ok and all(t < TAN_PASS for t in tans.values())
    label = "PASS" if ok else "KILL" if kill else "INCONCLUSIVE"
    if label != "PASS" and any(p["cand"]["undertrained"] for p in plats.values()):
        label = "INCONCLUSIVE-undertrained"
    return {"lines": lines, "label": label, "err20_ratio_per_train_seed": ratios, "tangential_over_radial": tans,
            "momentum_ratio_per_train_seed_report_only": moms, "plateau": plats}


ORDER = {"PASS": 0, "INCONCLUSIVE": 1, "INCONCLUSIVE-undertrained": 1, "KILL": 2}


def full_run_cells(res, arm):
    """Per train seed (lowest first) cells of one arm on one eval set, as in neighbour-context-pair-branch-full-run."""
    cells = {}
    for s in res["train_seeds"]:
        a, c = res["arms"][f"{arm}-s{s}"], res["arms"][f"ctrl-s{s}"]
        cells[s] = {"err20_ratio": a["err20_mean"] / c["err20_mean"], "tangential_over_radial": a["tangential_over_radial"],
                    "momentum_ratio": a["momentum_error_mean"] / c["momentum_error_mean"],
                    "over_frozen": a["err20_mean"] / res["arms"]["frozen-exp-e"]["err20_mean"],
                    "plateau": plateau(a["validation"], res["iters"])}
    return cells


def full_run_set_label(cells):
    mom = float(np.mean([c["momentum_ratio"] for c in cells.values()]))
    kill = any(c["err20_ratio"] >= ERR_KILL for c in cells.values()) or mom > 1.0
    ok = all(c["err20_ratio"] <= ERR_PASS and c["tangential_over_radial"] < TAN_PASS and c["over_frozen"] <= 1.0 for c in cells.values())
    label = "KILL" if kill else "PASS" if ok else "INCONCLUSIVE"
    return label, mom


def combine_full_run(held, fresh):
    out = {"lines": "full-run", "momentum_rule": "pooled over train seeds per eval set, KILL if above 1.0 on either set", "arms": {}}
    for arm in ("pen", "cand"):
        sets = {}
        for name, res in (("held-out", held), ("fresh", fresh)):
            cells = full_run_cells(res, arm)
            label, mom = full_run_set_label(cells)
            sets[name] = {"label": label, "momentum_ratio_pooled": mom, "cells": cells}
        label = max((v["label"] for v in sets.values()), key=lambda l: ORDER[l])
        if label != "PASS" and any(c["plateau"]["undertrained"] for v in sets.values() for c in v["cells"].values()):
            label = "INCONCLUSIVE-undertrained"
        out["arms"][arm] = {"label": label, "sets": sets}
    out["headline_arm"] = "pen"
    out["label"] = out["arms"]["pen"]["label"]
    out["control_err20_cv_across_eval_seeds"] = {"held-out": held["control_err20_spread"]["cv_across_eval_seeds"], "fresh": fresh["control_err20_spread"]["cv_across_eval_seeds"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", nargs=3, metavar=("HELD_OUT_JSON", "FRESH_JSON", "OUT"), default=None)
    ap.add_argument("--arm-cache-dir", default=None)
    ap.add_argument("--e-ckpt")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--train-dir")
    ap.add_argument("--train-seeds", default="4739")
    ap.add_argument("--eval-seeds", default="9100,9200,9300")
    ap.add_argument("--scenes", type=int, default=32)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lines", choices=["seed-replication", "tangential-penalty", "full-run"], default=None)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.combine:
        held, fresh, out_path = args.combine
        out = combine_full_run(json.load(open(held)), json.load(open(fresh)))
        json.dump(out, open(out_path + ".tmp", "w"), indent=1)
        os.replace(out_path + ".tmp", out_path)
        print("verdict", json.dumps(out), flush=True)
        return
    for need in ("e_ckpt", "ckpt_dir", "train_dir", "iters", "out"):
        if getattr(args, need) is None:
            raise SystemExit(f"--{need.replace('_', '-')} is required")
    dev = torch.device(args.device)
    seeds = [int(s) for s in args.train_seeds.split(",")]
    eseeds = [int(s) for s in args.eval_seeds.split(",")]
    data = {e: fd.base(args.scenes, STEPS, e, dev) for e in eseeds}
    arms = {"frozen-exp-e": (args.e_ckpt, STOCK, None)}
    for s in seeds:
        arms[f"ctrl-s{s}"] = (f"{args.ckpt_dir}/pairctx-ctrl-s{s}.pt", STOCK, f"{args.train_dir}/pairctx-ctrl-s{s}_train.json")
        arms[f"cand-s{s}"] = (f"{args.ckpt_dir}/pairctx-cand-s{s}.pt", CAND, f"{args.train_dir}/pairctx-cand-s{s}_train.json")
        if args.lines == "full-run":
            arms[f"pen-s{s}"] = (f"{args.ckpt_dir}/pairctx-pen-s{s}.pt", CAND, f"{args.train_dir}/pairctx-pen-s{s}_train.json")
    res = {"eval_seeds": eseeds, "train_seeds": seeds, "scenes": args.scenes, "ticks": STEPS, "iters": args.iters, "arms": {}}
    for name, (ck, var, tj) in arms.items():
        cache = f"{args.arm_cache_dir}/{name}.json" if args.arm_cache_dir else None
        if cache and os.path.exists(cache):
            res["arms"][name] = json.load(open(cache))
            print(name, "cached, skipping", flush=True)
            continue
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
        if cache:
            os.makedirs(args.arm_cache_dir, exist_ok=True)
            json.dump(a, open(cache + ".tmp", "w"))
            os.replace(cache + ".tmp", cache)
        print(name, {k: v for k, v in a.items() if k not in ("per_eval_seed", "validation")}, flush=True)
    c20 = [res["arms"][f"ctrl-s{s}"]["per_eval_seed"][str(e)]["err20"] for s in seeds for e in eseeds]
    res["control_err20_spread"] = {"cv_all_cells": float(np.std(c20) / np.mean(c20)),
                                   "cv_across_train_seeds": float(np.std([res["arms"][f"ctrl-s{s}"]["err20_mean"] for s in seeds]) / np.mean([res["arms"][f"ctrl-s{s}"]["err20_mean"] for s in seeds])),
                                   "cv_across_eval_seeds": float(np.mean([np.std([res["arms"][f"ctrl-s{s}"]["per_eval_seed"][str(e)]["err20"] for e in eseeds]) / np.mean([res["arms"][f"ctrl-s{s}"]["per_eval_seed"][str(e)]["err20"] for e in eseeds]) for s in seeds]))}
    res["frozen_ratio_cand_over_frozen"] = {str(s): res["arms"][f"cand-s{s}"]["err20_mean"] / res["arms"]["frozen-exp-e"]["err20_mean"] for s in seeds}
    if args.lines == "full-run":
        res["frozen_ratio_pen_over_frozen"] = {str(s): res["arms"][f"pen-s{s}"]["err20_mean"] / res["arms"]["frozen-exp-e"]["err20_mean"] for s in seeds}
        res["verdict"] = {"lines": "full-run", "note": "per-set cells; the verdict is printed by --combine on both sets"}
    else:
        res["verdict"] = score_lines(res, seeds, args.iters, args.lines) if args.lines else score(res, seeds, args.iters)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out + ".tmp", "w"), indent=1)
    os.replace(args.out + ".tmp", args.out)
    print("verdict", json.dumps(res["verdict"]), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
