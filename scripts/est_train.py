"""Train / load the node-pair error estimator for a kernel (analytic labels on uniform, flyby t0, flyby t1000 by default).

Usage: PYTHONPATH=. python scripts/est_train.py --kernel analytic --out checkpoints/est_analytic.pt
"""
import argparse
from types import SimpleNamespace

import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import kernels

NFEAT = 12


def make_kernel(name, dev, ckpt="checkpoints/gravity_central_v1.pt"):
    if name == "learned":
        return kernels.learned(ckpt, dev)
    return {"analytic": kernels.analytic, "inv_distance": kernels.inv_distance, "yukawa": kernels.yukawa}[name]()


def train_estimator(kernel, dev, states=("uniform", "flyby_t0", "flyby_t1000"), npz="results/flyby_100k.npz", bodies=100000, cap=8,
                    n_pairs=40000, n_big=4000, epochs=40, label_thetas=(0.35, 0.7, 1.2), seed=4738):
    kernels.current = kernel
    gen = torch.Generator(device=dev).manual_seed(seed)
    args = SimpleNamespace(npz=npz, bodies=bodies)
    xs, ys = [], []
    for s in states:
        pos = ao.load_state(s, args, dev)
        tr = ao.Tree(pos, 18)
        fl = dt.Flat(tr)
        ga, gb = de.collect_pairs(tr, fl, list(label_thetas), cap, n_pairs, n_big, gen)
        y, _, _ = de.label(tr, fl, ga, gb, gen)
        x = de.pair_feats(fl, ga, gb)
        keep = torch.isfinite(x).all(1) & torch.isfinite(y)
        xs.append(x[keep].cpu())
        ys.append(y[keep].cpu())
    x, y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    nv = len(x) // 10
    est = de.train(x[perm[nv:]], y[perm[nv:]], x[perm[:nv]], y[perm[:nv]], [0.9], dev, epochs, seed)
    return est


def save(est, path, kernel_name):
    torch.save({"state": est.state_dict(), "kernel": kernel_name}, path)


def load(path, dev):
    blob = torch.load(path, map_location=dev)
    est = de.Est(torch.zeros(NFEAT), torch.ones(NFEAT), 2).to(dev)
    est.load_state_dict(blob["state"])
    return de.Head(est.eval(), 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="analytic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    dev = torch.device("cuda")
    k = make_kernel(a.kernel, dev)
    save(train_estimator(k, dev, epochs=a.epochs), a.out, a.kernel)
    print("saved", a.out)
