"""Train node-pair error estimators (est_train logic, labels kept) for the interpretability test.
Usage: PYTHONPATH=. python scripts/interp_train.py --kernels analytic,inv_distance,yukawa30,plw0.5,plw1.5,plw3
Writes checkpoints/interp_est_<kernel>[_s<seed>].pt and results/interp_labels_<kernel>[_s<seed>].npz (x, y, state id)."""
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import est_train
from scripts import interp_kernels as ik
from scripts import kernels

STATES = ("uniform", "flyby_t0", "flyby_t1000")


def build(kernel, dev, seed, epochs, npz="results/flyby_100k.npz", bodies=100000, cap=8, n_pairs=40000, n_big=4000, thetas=(0.35, 0.7, 1.2)):
    kernels.current = kernel
    gen = torch.Generator(device=dev).manual_seed(seed)
    args = SimpleNamespace(npz=npz, bodies=bodies)
    xs, ys, sid = [], [], []
    for si, s in enumerate(STATES):
        pos = ao.load_state(s, args, dev)
        tr = ao.Tree(pos, 18)
        fl = dt.Flat(tr)
        ga, gb = de.collect_pairs(tr, fl, list(thetas), cap, n_pairs, n_big, gen)
        y, _, _ = de.label(tr, fl, ga, gb, gen)
        x = de.pair_feats(fl, ga, gb)
        keep = torch.isfinite(x).all(1) & torch.isfinite(y)
        xs.append(x[keep].cpu())
        ys.append(y[keep].cpu())
        sid.append(torch.full((int(keep.sum()),), si))
    x, y, sid = torch.cat(xs), torch.cat(ys), torch.cat(sid)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    nv = len(x) // 10
    est = de.train(x[perm[nv:]], y[perm[nv:]], x[perm[:nv]], y[perm[:nv]], [0.9], dev, epochs, seed)
    with torch.no_grad():
        pv = est.net(((x[perm[:nv]].to(dev) - est.mu) / est.sd).float())[:, 0].double().cpu()
    val = {"val_rmse": float((pv - y[perm[:nv]]).pow(2).mean().sqrt()), "val_std_y": float(y[perm[:nv]].std()), "n": len(x)}
    return est, x, y, sid, val


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="analytic,inv_distance,yukawa30,plw0.5,plw1.5,plw3")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="results/interp_train.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    sfx = "" if a.seed == 4738 else f"_s{a.seed}"
    for name in a.kernels.split(","):
        t0 = time.time()
        est, x, y, sid, val = build(ik.get(name), dev, a.seed, a.epochs)
        est_train.save(est, f"checkpoints/interp_est_{name}{sfx}.pt", name)
        np.savez(f"results/interp_labels_{name}{sfx}.npz", x=x.numpy(), y=y.numpy(), sid=sid.numpy())
        val["time_s"] = time.time() - t0
        res[name] = val
        print(name, json.dumps(val), flush=True)
        json.dump(res, open(a.out, "w"))
