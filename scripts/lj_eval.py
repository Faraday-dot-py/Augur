"""Metrics for the LJ learned-force model vs truth: learned f(d) vs analytic, per-scene position error over frames
(with the truth-rerun noise floor), and time-averaged T, P, E/N, g(r), psi6 over the last frames."""
import argparse
import json

import numpy as np
import torch

from model.lj_force import LJForceDynamics
from model.token_graph import build_radius_graph_periodic
from scripts import lj_sim

BINS = np.arange(0.0, 5.0001, 0.05)
NN_CUTOFF = 1.6  # first coordination shell scale (see docs/debugging/experiment-log.md, LJ scaling entry)


def pair_dist(pos, box):
    rel = pos[:, None, :] - pos[None, :, :]
    rel = rel - box * torch.round(rel / box)
    return rel, torch.sqrt((rel ** 2).sum(-1))


def gr_psi6(frames, box, dev):
    box_t = torch.tensor(box, device=dev, dtype=torch.float64)
    n = frames.shape[1]
    rho = n / float(box[0] * box[1])
    hist = torch.zeros(len(BINS) - 1, device=dev, dtype=torch.float64)
    edges = torch.tensor(BINS, device=dev, dtype=torch.float64)
    psi = []
    for f in frames:
        p = torch.tensor(f, device=dev, dtype=torch.float64)
        rel, d = pair_dist(p, box_t)
        d.fill_diagonal_(1e9)
        hist += torch.histc(d[d < BINS[-1]], bins=len(BINS) - 1, min=0.0, max=float(BINS[-1]))
        nn = torch.topk(d, 6, largest=False).indices
        v = rel[torch.arange(n, device=dev)[:, None], nn]
        th = torch.atan2(v[..., 1], v[..., 0])
        psi.append(float(torch.abs(torch.exp(6j * th).mean(1)).mean()))
    ring = np.pi * (BINS[1:] ** 2 - BINS[:-1] ** 2)
    g = hist.cpu().numpy() / (len(frames) * n * rho * ring)
    return g.tolist(), float(np.mean(psi))


def gr_psi6_scalable(frames, box, dev, nn_cutoff=NN_CUTOFF):
    """Same g(r) and psi6 as `gr_psi6`, but via the O(N) periodic cell-list (model/token_graph) instead of
    a dense N x N distance matrix -- needed once N is large enough that N^2 float64 doesn't fit (see the LJ
    scaling entry in docs/debugging/experiment-log.md, job 3096 OOM at N=96768). psi6 here averages over
    neighbours within `nn_cutoff` (first coordination shell) rather than the strict 6 nearest; the two
    definitions agree closely in a dense solid/liquid, where each shell holds close to 6 neighbours."""
    box_t = torch.tensor(box, device=dev, dtype=torch.float64)
    n = frames.shape[1]
    rho = n / float(box[0] * box[1])
    hist = torch.zeros(len(BINS) - 1, device=dev, dtype=torch.float64)
    psi = []
    for f in frames:
        p = torch.tensor(f, device=dev, dtype=torch.float64)
        edges = build_radius_graph_periodic(p, box_t, float(BINS[-1]))
        src, dst = edges[0], edges[1]
        rel = p[dst] - p[src]
        rel = rel - box_t * torch.round(rel / box_t)
        d = torch.sqrt((rel ** 2).sum(-1) + 1e-12)
        hist += torch.histc(d[d < BINS[-1]], bins=len(BINS) - 1, min=0.0, max=float(BINS[-1]))
        keep = d <= nn_cutoff
        e6 = torch.exp(6j * torch.atan2(rel[keep, 1], rel[keep, 0]).to(torch.complex128))
        sums = torch.zeros(n, dtype=torch.complex128, device=dev).index_add_(0, dst[keep], e6)
        counts = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, dst[keep], torch.ones(int(keep.sum()), dtype=torch.float64, device=dev))
        valid = counts > 0
        if valid.any():
            psi.append(float((sums[valid] / counts[valid]).abs().mean()))
    ring = np.pi * (BINS[1:] ** 2 - BINS[:-1] ** 2)
    g = hist.cpu().numpy() / (len(frames) * n * rho * ring)
    return g.tolist(), float(np.mean(psi)) if psi else 0.0


def err_curve(a, b, box):
    d = a - b
    d = d - box * np.round(d / box)
    return np.linalg.norm(d, axis=-1).mean(-1)


def summary(d, dev, tail):
    st = d["stats"][-tail:]
    g, psi = gr_psi6(d["pos"][-tail::max(1, tail // 10)], d["box"], dev)
    return {"T": float(st[:, 1].mean()), "P": float(st[:, 4].mean()), "E": float(st[:, 3].mean()), "psi6": psi, "g": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/lj_force.pt")
    ap.add_argument("--scenes", default=",".join(lj_sim.SCENES))
    ap.add_argument("--tail", type=int, default=200)
    ap.add_argument("--prefix", default="results/lj")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/lj_eval.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=dev)
    model = LJForceDynamics(dt=ck["args"]["dt"], substeps=20, width=ck["args"].get("width", 64)).double().to(dev)
    model.load_state_dict(ck["state"])
    out = {"iter": ck["iter"]}
    d = torch.linspace(0.75, 2.5, 500, dtype=torch.float64, device=dev)
    with torch.no_grad():
        f = model.pair_force(d).cpu().numpy()
    dn = d.cpu().numpy()
    _, f_rc = lj_sim._sf_consts(lj_sim.RC)
    ft = 24 * (2 * dn ** -13 - dn ** -7) - f_rc
    out["force"] = {"rel_l2": float(np.linalg.norm(f - ft) / np.linalg.norm(ft)),
                    "max_abs_tail_1.2_2.5": float(np.abs(f - ft)[dn > 1.2].max()),
                    "max_abs_core_0.75_1.2": float(np.abs(f - ft)[dn <= 1.2].max()),
                    "d": dn[::10].tolist(), "learned": f[::10].tolist(), "true": ft[::10].tolist()}
    for name in args.scenes.split(","):
        truth = np.load(f"{args.prefix}_{name}.npz")
        model_r = np.load(f"{args.prefix}_model{args.tag}_{name}.npz")
        rerun = np.load(f"{args.prefix}_rerun{args.tag}_{name}.npz")
        box = truth["box"]
        ks = [1, 2, 5, 10, 20, 50, 100]
        nf = len(rerun["pos"])
        out[name] = {
            "err_model": {k: float(err_curve(model_r["pos"][k], truth["pos"][k], box).mean()) for k in ks if k < nf},
            "err_rerun_floor": {k: float(err_curve(rerun["pos"][k], truth["pos"][k], box).mean()) for k in ks if k < nf},
            "truth": summary(truth, dev, args.tail),
            "model": summary(model_r, dev, args.tail),
            "model_energy_series": model_r["stats"][::40, 3].tolist(),
            "truth_energy_series": truth["stats"][::40, 3].tolist(),
        }
        print(name, {k: out[name][k] for k in ("err_model", "err_rerun_floor")}, flush=True)
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
