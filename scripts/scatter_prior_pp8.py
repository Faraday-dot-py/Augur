"""Ablation of the scatter-field gap fix (prior-residual kernel + exact pair complement, pp 2->8, knn 48) on Exp E data.
Every model is trained from scratch with the identical loop (Exp E data, seed 4738, batch 16, k 4->20, lr 1e-3 cosine, same iterations)
and evaluated on the Exp E held-out sets (seeds 9000/12000, 48 scenes: first 24 gs scenes 10-200 bodies, last 24 clusters of 80-300 bodies).
Models: ctl (frozen architecture, matched iterations), prior (fix 1 alone), pp8 (fix 2 alone), pair (1+2), cf (CentralForceDynamics-equivalent CFDense).

Usage: PYTHONPATH=. python scripts/scatter_prior_pp8.py --models ctl,pair,prior,pp8,cf --iters 4000
       PYTHONPATH=. python scripts/scatter_prior_pp8.py --models pair --eval-ckpt <ckpt>   # eval only (also any other checkpoint via --variant)"""
import argparse
import json
import os
from types import SimpleNamespace

import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as ts
from scripts.gap_probe import CFDense, force_eval

EPS = 0.5
DT = 0.1
CFG = ts.EXPS["E"]
VARIANT = {"ctl": "ms_kp_pot_v_g128", "prior": "ms_kp_pot_v_g128_prior", "pp8": "ms_kp_pot_v_g128_pp8", "pair": "ms_kp_pot_v_g128_prior_pp8"}
N_AB = 24


def build(name, variant=None):
    if name == "cf":
        return CFDense()
    return ts.build(CFG, variant or VARIANT[name], DT)


def head(m):
    return [round(m["err"][i], 5) for i in (4, 9, 19, 49, 99)]


def evaluate(model, dev):
    res = {}
    for seed in ts.EVAL_SEEDS:
        data = ts.make_data(CFG, 48, 100, seed, dev, DT, EPS)
        for tag, sub in (("ab", data[:N_AB]), ("cl", data[N_AB:])):
            m, _ = ts.evaluate_model(model, sub, dev, DT, EPS)
            P, V, M, mask = sf.to_tensors(sub, dev)
            r = {"err": m["err"], "dE_model": m["energy_drift_rel_model"][100], "dE_true": m["energy_drift_rel_true"][100],
                 "dP_model": m["mom_drift_model"][100], "dL_model": m["angmom_drift_rel_model"][100],
                 "force": force_eval(model, P, V, M, mask)}
            res[f"{tag}_{seed}"] = r
            print(f"{tag} seed {seed} err@5/10/20/50/100 {head(m)} dE/E {r['dE_model']:.4g} (true {r['dE_true']:.4g}) force rel {r['force']['rel_rms']:.4f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ctl,pair,prior,pp8,cf")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--out", default="results/scatter_pp8")
    ap.add_argument("--ckpt", default="checkpoints/scatter_pp8")
    ap.add_argument("--eval-ckpt", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    dev = torch.device("cuda")
    tensors = None
    for name in args.models.split(","):
        key = f"{name}{args.tag}_{args.iters}" if not args.eval_ckpt else f"{name}{args.tag}_evalonly"
        out_path = f"{args.out}/{key}.json"
        if os.path.exists(out_path):
            print("skip", key, flush=True)
            continue
        torch.manual_seed(4738)
        model = build(name, args.variant).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        r = {"model": name, "variant": args.variant or VARIANT.get(name), "params": n_params}
        if args.eval_ckpt:
            model.load_state_dict(torch.load(args.eval_ckpt, map_location=dev, weights_only=False)["model"])
            r["ckpt"] = args.eval_ckpt
        else:
            if tensors is None:
                tensors = sf.to_tensors(ts.make_data(CFG, args.train, 30, 4738, dev, DT, EPS), dev)
            targs = SimpleNamespace(lr=1e-3, time_budget=0.0, iters=args.iters, k_start=4, k_end=20, batch=16, steps=30, device=str(dev),
                                    seed=4738, log_every=100, opt="")
            n_it, secs = ts.train(model, tensors, targs, f"{args.ckpt}/{name}{args.tag}.pt", lambda s: print(name, s, flush=True))
            r.update(train_iters=n_it, train_seconds=secs)
        model.eval()
        r["eval"] = evaluate(model, dev)
        json.dump(r, open(out_path, "w"))
        print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
