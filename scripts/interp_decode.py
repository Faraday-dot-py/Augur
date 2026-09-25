"""H7 interpretability, task 2: recover the force law of a black-box kernel from (a) direct pair-force samples (baseline), (b) coarse/fine
discrepancy of node pairs (vector, or scalar relative magnitude), (c) the trained estimator's predicted labels.
Parametric decoder f(d) = G d (d^2 + eps^2)^-((p+1)/2) exp(-kappa d); fits by Adam, best of 4 starts (batched).
Usage: PYTHONPATH=. python scripts/interp_decode.py --out results/interp_decode.json"""
import argparse
import json

import numpy as np
import torch

from scripts import est_train
from scripts import interp_kernels as ik
from scripts import interp_stage1 as s1
from scripts import interp_synth as sy
from scripts import kernels

TRUE = {"plw0.5": (0.5, 0.5, 0.0), "plw1": (1.0, 0.5, 0.0), "plw1.5": (1.5, 0.5, 0.0), "plw2": (2.0, 0.5, 0.0), "plw3": (3.0, 0.5, 0.0), "yukawa30": (2.0, 0.5, 1 / 30)}
P0 = [0.7, 1.5, 2.5, 3.5]


def fth(d, th, full):
    logG, p, leps, kr = th
    e2 = (torch.exp(leps) ** 2)[:, None] if full else 0.25
    scr = torch.exp(-(kr ** 2)[:, None] * d) if full else 1.0
    return torch.exp(logG)[:, None] * d * (d ** 2 + e2) ** (-(p[:, None] + 1) / 2) * scr


def force_vec(d, th, full):
    """d (M,...,2) -> (S,M,...,2)"""
    dn = d.norm(dim=-1).clamp(min=1e-9)
    shape = dn.shape
    f = fth(dn.reshape(1, -1), th, full).reshape(len(th[0]), *shape)
    return f[..., None] * (d / dn[..., None])[None]


def init_th(dev, full, G0=True):
    S = len(P0)
    logG = torch.zeros(S, dtype=torch.float64, device=dev)
    p = torch.tensor(P0, dtype=torch.float64, device=dev)
    leps = torch.full((S,), float(np.log(0.3)), dtype=torch.float64, device=dev)
    kr = torch.full((S,), 0.05, dtype=torch.float64, device=dev)
    return [t.clone().requires_grad_(True) for t in (logG, p, leps, kr)]


def run_fit(lossfn, dev, full, use_G, iters=500):
    th = init_th(dev, full)
    params = [th[0], th[1]] + ([th[2], th[3]] if full else [])
    if not use_G:
        params = params[1:]
    opt = torch.optim.Adam(params, lr=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        opt.zero_grad()
        l = lossfn(th).sum()
        l.backward()
        opt.step()
        sch.step()
        with torch.no_grad():
            th[1].clamp_(0.05, 6.0)
    with torch.no_grad():
        l = lossfn(th)
        b = int(torch.nan_to_num(l, nan=1e30).argmin())
        return {"p": float(th[1][b]), "eps": float(torch.exp(th[2][b])) if full else 0.5, "kappa": float(th[3][b] ** 2) if full else 0.0,
                "G": float(torch.exp(th[0][b])) if use_G else 1.0, "loss": float(l[b])}


def node_data(kern, N, gen, dev, n=16):
    dist = torch.exp(torch.rand(N, device=dev, generator=gen, dtype=torch.float64) * np.log(100) + np.log(1.5))
    rho = torch.exp(torch.rand(N, device=dev, generator=gen, dtype=torch.float64) * np.log(12) + np.log(0.05))
    s = rho * dist
    a = sy.square_cloud(N, n, s, gen, dev)
    b = sy.square_cloud(N, n, s, gen, dev) + torch.stack([dist, torch.zeros_like(dist)], 1)[:, None]
    ph = torch.rand(N, device=dev, generator=gen, dtype=torch.float64) * 6.2831853
    R = sy.rot(ph)
    a, b = a @ R.transpose(1, 2), b @ R.transpose(1, 2)
    d = b[:, None] - a[:, :, None]
    fine = kern.pair(d.reshape(-1, 2)).view(N, n, n, 2).sum((1, 2))
    r = b.mean(1) - a.mean(1)
    coarse = n * n * kern.pair(r)
    return a, b, fine - coarse, coarse


def loss_vec(a, b, delta, coarse, full):
    n = a.shape[1]
    d = b[:, None] - a[:, :, None]
    rr = b.mean(1) - a.mean(1)

    def lf(th):
        fine = force_vec(d, th, full).sum((2, 3))
        c = n * n * force_vec(rr, th, full)
        dl = fine - c
        return ((dl - delta[None]) ** 2).sum(-1).div((delta ** 2).sum(-1)[None] + 1e-30).mean(1)
    return lf


def loss_scal(a, b, delta, coarse, full):
    n = a.shape[1]
    d = b[:, None] - a[:, :, None]
    rr = b.mean(1) - a.mean(1)
    obs = torch.log(delta.norm(dim=-1) / coarse.norm(dim=-1))

    def lf(th):
        fine = force_vec(d, th, full).sum((2, 3))
        c = n * n * force_vec(rr, th, full)
        m = torch.log((fine - c).norm(dim=-1) / c.norm(dim=-1))
        return ((m - obs[None]) ** 2).mean(1)
    return lf


def loss_pair(dq, fobs, full):
    lo = torch.log(fobs.abs())

    def lf(th):
        f = fth(dq[None], th, full)
        return ((torch.log(f.abs() + 1e-300) - lo[None]) ** 2).mean(1)
    return lf


def relerr(fit, kt):
    p, e, k = TRUE[kt]
    return {"p": fit["p"], "p_relerr": abs(fit["p"] - p) / p, "eps": fit["eps"], "kappa": fit["kappa"], "loss": fit["loss"], "G": fit["G"]}


def decode_all(dev, out):
    Ns, sigs, seeds = [4, 16, 64, 256], [0.0, 0.01], [4738, 9000, 12000]
    for kt in TRUE:
        p, e, k = TRUE[kt]
        kern = ik.plw(p, e, 1 / k if k else None)
        kernels.current = kern
        rec = {}
        for N in Ns:
            for sg in sigs:
                for meth in ("pair_p", "pair_full", "vec_p", "vec_full", "scal_p", "scal_full"):
                    rs = []
                    for sd in seeds:
                        gen = torch.Generator(device=dev).manual_seed(sd)
                        full = meth.endswith("full")
                        if meth.startswith("pair"):
                            dq = torch.exp(torch.rand(N, device=dev, generator=gen, dtype=torch.float64) * np.log(500) + np.log(0.3))
                            fo = kern.f(dq) * (1 + sg * torch.randn(N, device=dev, generator=gen, dtype=torch.float64))
                            fit = run_fit(loss_pair(dq, fo, full), dev, full, True, 1500)
                        else:
                            a, b, dl, c = node_data(kern, N, gen, dev)
                            dl = dl * (1 + sg * torch.randn_like(dl))
                            if meth.startswith("vec"):
                                fit = run_fit(loss_vec(a, b, dl, c, full), dev, full, True)
                            else:
                                fit = run_fit(loss_scal(a, b, dl, c, full), dev, full, False)
                        rs.append(relerr(fit, kt))
                    rec[f"{meth}|N{N}|s{sg}"] = {"p_relerr_median": float(np.median([r["p_relerr"] for r in rs])), "p_relerr_max": float(max(r["p_relerr"] for r in rs)),
                                                "eps_med": float(np.median([r["eps"] for r in rs])), "kappa_med": float(np.median([r["kappa"] for r in rs])),
                                                "p_med": float(np.median([r["p"] for r in rs]))}
            print(kt, N, flush=True)
            out["decode"][kt] = rec
            json.dump(out, open(a_out, "w"))


def est_route(dev, out):
    gen = torch.Generator(device=dev).manual_seed(4738)
    for kt, en in (("plw0.5", "plw0.5"), ("plw1", "inv_distance"), ("plw1.5", "plw1.5"), ("plw2", "analytic"), ("plw3", "plw3"), ("yukawa30", "yukawa30")):
        head = est_train.load(f"checkpoints/interp_est_{en}.pt", dev)
        d = np.load(f"results/interp_labels_{en}.npz")
        lab = {"x": d["x"], "y": d["y"]}
        Pmax = 400
        rho, dist = s1.joint_sample(lab, Pmax, gen, dev)
        a, b, s = s1.pair_clouds(Pmax, rho, dist, gen, dev)
        yest = s1.pred(head, sy.feats(a, b, s, s, s1.N_PTS))
        p0, e0, k0 = TRUE[kt]
        true_k = ik.plw(p0, e0, 1 / k0 if k0 else None)
        kernels.current = true_k
        ytrue = sy.label(true_k, a, b)
        rec = {"est_vs_truth_rmse": float((yest - ytrue).pow(2).mean().sqrt()), "est_bias": float((yest - ytrue).mean())}
        if kt == "yukawa30":
            grid = [(2.0, 0.5, kap) for kap in np.linspace(0, 0.1, 41)]
            axname = "kappa"
        else:
            grid = [(pp, ee, 0.0) for pp in np.arange(0.2, 4.01, 0.1) for ee in (0.25, 0.35, 0.5, 0.7, 1.0)]
            axname = "p_eps"
        cache = []
        for (pp, ee, kk) in grid:
            kg = ik.plw(pp, ee, 1 / kk if kk > 0 else None)
            kernels.current = kg
            cache.append(sy.label(kg, a, b))
        L = torch.stack(cache)
        for tag, y in (("est", yest), ("truelabel", ytrue)):
            for M in (5, 10, 25, 100, 400):
                for sub in ("p_only_eps_known", "p_eps"):
                    if kt == "yukawa30" and sub == "p_eps":
                        continue
                    idx = torch.arange(M, device=dev)
                    valid = torch.tensor([abs(g[1] - 0.5) < 1e-9 or sub == "p_eps" for g in grid], device=dev)
                    mse = (L[:, idx] - y[None, idx]).pow(2).mean(1)
                    mse[~valid] = 1e30
                    j = int(mse.argmin())
                    g = grid[j]
                    rec[f"{tag}|M{M}|{sub}"] = {"p": float(g[0]), "eps": float(g[1]), "kappa": float(g[2]), "mse": float(mse[j])}
        rec["grid_axis"] = axname
        out["est_route"][kt] = rec
        print("est_route", kt, flush=True)
        json.dump(out, open(a_out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/interp_decode.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    a_out = a.out
    dev = torch.device("cpu" if a.smoke else "cuda")
    out = {"decode": {}, "est_route": {}}
    if a.only != "est":
        decode_all(dev, out)
    if a.only != "decode":
        est_route(dev, out)
