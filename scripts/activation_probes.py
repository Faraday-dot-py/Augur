"""Linear probes on scatter-field activations (curiosity, no pass/kill): law identity (8-class softmax regression), distance (ridge R^2 on log r for the
pair MLP, on radius from centre for the mesh net), cell mass (ridge R^2 on log cell mass, mesh only; the pair MLP has no mass input). Held-out by scene
(mesh) or every 7th radius (pair). Shuffled-label control gives chance level.
Usage: PYTHONPATH=. python scripts/activation_probes.py --ckpt ckpt/main_s4738.pt --out results/activation_probes_s4738.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from scripts import scatter_multi as sm
from scripts.sae_interp import LAWS, mesh_acts, pair_acts

MESH_LAYERS = ("enc0", "enc2", "enc4", "dec3", "dec1", "dec0")
PAIR_LAYERS = ("l1", "l2")


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    A = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=Xtr.device)], 1)
    B = torch.cat([Xte, torch.ones(len(Xte), 1, device=Xte.device)], 1)
    I = torch.eye(A.shape[1], device=A.device)
    w = torch.linalg.solve(A.T @ A + lam * I, A.T @ ytr)
    return float(1 - ((B @ w - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def softmax_acc(Xtr, ytr, Xte, yte, k, steps=500):
    W = torch.zeros(Xtr.shape[1], k, device=Xtr.device, requires_grad=True)
    b = torch.zeros(k, device=Xtr.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-4 * (W ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((Xte @ W + b).argmax(-1) == yte).float().mean())


def probe_set(X, lab, cont, grp, dev, seed):
    te = grp == grp.max()
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
    Xn = ((X - mu) / sd).to(dev)
    tr, te = (~te).to(dev), te.to(dev)
    lab = lab.to(dev)
    out = {"law_acc": softmax_acc(Xn[tr], lab[tr], Xn[te], lab[te], len(LAWS)), "law_chance": 1 / len(LAWS)}
    perm = torch.randperm(int(tr.sum()), generator=torch.Generator().manual_seed(seed)).to(dev)
    out["law_acc_shuffled_label_control"] = softmax_acc(Xn[tr], lab[tr][perm], Xn[te], lab[te], len(LAWS))
    for name, y in cont.items():
        y = y.to(dev)
        out[f"{name}_r2"] = ridge_r2(Xn[tr], y[tr], Xn[te], y[te])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--cells", type=int, default=400)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    model = sm.build().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    out = {"ckpt": a.ckpt, "seed": a.seed, "mesh": {}, "pair": {}}
    for layer in MESH_LAYERS:
        X, lab, cont, sid = mesh_acts(model, dev, a.scenes, a.cells, 3, a.seed, layer)
        out["mesh"][layer] = probe_set(X, lab, cont, sid % a.scenes, dev, a.seed)
        print("mesh", layer, out["mesh"][layer], flush=True)
    for layer in PAIR_LAYERS:
        X, lab, cont, grp = pair_acts(model, dev, (0.3, 1.0, 3.0), 1000, layer)
        out["pair"][layer] = probe_set(X, lab, cont, grp, dev, a.seed)
        print("pair", layer, out["pair"][layer], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
