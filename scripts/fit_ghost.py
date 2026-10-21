"""Fit the learned ghost-cell closure (ScatterField gpad, 9 weights) on the kernel potential with labels from a larger domain.
For each training scene the kernel potential is computed on the doubled (2G) FFT domain; the label for the ghost cell one step beyond each side of the grid is the
exact kernel potential there (phi[G], phi[-1]), the inputs are the outermost 3 cells (3 deep x 3 along the edge). Scenes: open-domain clusters (bodies crossing the box,
mass outside the box does not enter the grid density) at several ticks of the exact trajectories, plus clipped Gaussians. Least squares (convex), base weights frozen.
Writes a checkpoint {"model": state_dict} usable with variant ms_kp_pot_v_g128_ghost, and reports train / held-out residuals."""
import argparse
import json

import numpy as np
import torch

from scripts import edge_ghost_diag as dg
from scripts import scatter_edge_data as sed
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf


def collect(m, scenes, dev):
    Xs, Ys = [], []
    for pos in scenes:
        full = dg.kernel_phi(m, pos, dev)
        X, Y = dg.strips(full, m.grid)
        sc = Y.abs().max().clamp_min(1e-9)
        Xs.append(X / sc)
        Ys.append(Y / sc)
    return torch.cat(Xs), torch.cat(Ys)


def open_positions(seed, num, ticks, dev):
    data = sed.make_open_dataset(num, max(ticks), seed, dev)
    out = []
    for P, V in data:
        for t in ticks:
            p = P[t] - sf.CENTER
            out.append(p[(np.abs(p) < 32).all(1)])
    return out


def gauss_positions(seed, count):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = int(rng.choice([300, 2000, 20000]))
        out.append(np.clip(rng.normal(0, rng.uniform(5, 14), (n, 2)), -29, 29))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    args = ap.parse_args()
    dev = torch.device("cuda")
    m = dg.build(args.ckpt, dev)
    ticks = (0, 10, 20, 30, 40, 50)
    tr = open_positions(args.seed, 200, ticks, dev) + gauss_positions(args.seed, 12)
    ho = open_positions(9000, 40, ticks, dev) + gauss_positions(9000, 6)
    X, Y = collect(m, tr, dev)
    Xh, Yh = collect(m, ho, dev)
    w = torch.linalg.lstsq(X.double(), Y.double()[:, None]).solution[:, 0].float()
    res = {"train_rel_resid": float((X @ w - Y).abs().mean() / Y.abs().mean()), "heldout_rel_resid": float((Xh @ w - Yh).abs().mean() / Yh.abs().mean()),
           "w": w.view(3, 3).tolist(), "n_train_samples": int(len(Y)), "n_scenes": len(tr)}
    print("FIT", json.dumps(res), flush=True)
    m.gpad.weight.data = w.view(1, 1, 3, 3).clone()
    torch.save({"model": m.state_dict(), "it": 0}, args.out)
    json.dump(res, open(args.json, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
