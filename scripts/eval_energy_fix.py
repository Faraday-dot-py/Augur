"""Held-out comparison for the energy fixes. For each model spec: (a) state-space
rollouts from the true initial state on held-out seeds (n=20, 4 balls and 100
balls in n=100): err@5/10/20, energy vs truth at steps 20/50/100, total-momentum
error, non-finite; (b) single-ball wall curve (y-wall restitution, x-floor dE)
and head-on pair curve (relKE, relvy ratio, pass-through, momentum error) at
speeds 3..80 (training max 40).

Usage: PYTHONPATH=. python3 scripts/eval_energy_fix.py --models soupb=soupb:checkpoints/token_model_soup_b.pt \
    cons=cons:checkpoints/cons_pure.pt --seeds 9000 12000 --out results/eval_energy_fix.json
Spec: name=kind:ckpt[:adaptive_radius[:neighbor_radius]]
"""
import argparse
import json
import random

import numpy as np
import torch

from model.dataset import make_scenario_uniform
import scripts.energy_probe2 as p2
from scripts.ball_1k_rollout import truth
from scripts.energy_fix_common import build_model
from scripts.energy_probe import energy

DT, G = 0.15, 9.0
SPEEDS = [3, 6, 10, 15, 20, 30, 40, 60, 80]


def rollout_metrics(kind, ckpt, kw, device, seeds, nb, n, steps=100, per_seed=12):
    model = build_model(kind, ckpt, n, device, **kw)
    model.dynamics.cell_graph = nb > 50
    errs, es, ets, moms, finite = [], [], [], [], []
    for base in seeds:
        for i in range(per_seed):
            balls = make_scenario_uniform(nb, n, 2.3, random.Random(base + i))
            tp, tv = truth(balls, n, steps, DT, G, 0.75, 400.0, 8, device)
            pos, vel = torch.from_numpy(tp[0]).to(device), torch.from_numpy(tv[0]).to(device)
            hid = torch.zeros(nb, model.dynamics.hidden_dim, device=device)
            e, m, er, ok = [], [], [], True
            with torch.no_grad():
                for t in range(steps + 1):
                    tpt, tvt = torch.from_numpy(tp[t]).to(device), torch.from_numpy(tv[t]).to(device)
                    er.append(float((pos - tpt).norm(dim=1).mean()))
                    e.append(float(energy(pos, vel, G).mean()))
                    m.append(float((vel.sum(0) - tvt.sum(0)).norm()))
                    if t < steps:
                        pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
                        ok = ok and bool(torch.isfinite(pos).all() and torch.isfinite(vel).all())
            errs.append(er)
            es.append(e)
            moms.append(m)
            ets.append([float(energy(torch.from_numpy(tp[t]), torch.from_numpy(tv[t]), G).mean()) for t in range(steps + 1)])
            finite.append(ok)
    er, e, et, m = (np.nanmedian(np.array(a), axis=0) for a in (errs, es, ets, moms))
    mean_err = np.nanmean(np.array(errs), axis=0)
    return {"err_mean": {str(t): float(mean_err[t]) for t in (1, 5, 10, 20, 50, 100)},
            "err_median": {str(t): float(er[t]) for t in (1, 5, 10, 20, 50, 100)},
            "E_model": {str(t): float(e[t]) for t in (0, 20, 50, 100)},
            "E_truth": {str(t): float(et[t]) for t in (0, 20, 50, 100)},
            "mom_err": {str(t): float(m[t]) for t in (5, 20, 50, 100)},
            "finite_frac": float(np.mean(finite)), "n_scenes": len(errs)}


def wall_pair(kind, ckpt, kw, device):
    model = build_model(kind, ckpt, 800, device, **kw)
    p2.SPEEDS = SPEEDS
    wall = p2.wall_curve(model, device)
    pair = p2.pair_curve(model, device)
    return {"wall": wall, "pair": pair}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9000, 12000])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/eval_energy_fix.json")
    args = ap.parse_args()
    out = {}
    for spec in args.models:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        kind, ckpt = parts[0], parts[1]
        kw = {}
        if len(parts) > 2:
            kw["adaptive_radius"] = parts[2]
        if len(parts) > 3:
            kw["neighbor_radius"] = float(parts[3])
        r = {"n20_b4": rollout_metrics(kind, ckpt, kw, args.device, args.seeds, 4, 20),
             "n100_b100": rollout_metrics(kind, ckpt, kw, args.device, args.seeds, 100, 100, per_seed=3)}
        r.update(wall_pair(kind, ckpt, kw, args.device))
        out[name] = r
        print("==", name, flush=True)
        for k in ("n20_b4", "n100_b100"):
            print(k, json.dumps(r[k]), flush=True)
        for row in r["wall"]:
            print("wall", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
        for row in r["pair"]:
            print("pair", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    json.dump(out, open(args.out, "w"), default=float)
    print("done")


if __name__ == "__main__":
    main()
