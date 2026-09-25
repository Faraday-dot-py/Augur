"""H7 interpretability, tasks 1 and 4: what the node-pair estimator encodes about the force law (rho exponent, node anisotropy,
distance vs length scale) against exact synthetic labels and second-order Taylor theory; exchange / rotation / scale-covariance checks.
Usage: PYTHONPATH=. python scripts/interp_stage1.py --out results/interp_stage1.json"""
import argparse
import json

import numpy as np
import torch

from scripts import dual_estimator as de
from scripts import est_train
from scripts import interp_kernels as ik
from scripts import interp_synth as sy
from scripts import kernels

EPS = de.EPS
N_PTS = 64
KERNELS = ["analytic", "inv_distance", "yukawa30", "plw0.5", "plw1.5", "plw3"]


def pred(head, x):
    with torch.no_grad():
        return head(x)


def fit_slope(xv, yv):
    return float(np.polyfit(xv, yv, 1)[0])


def joint_sample(lab, P, gen, dev):
    x = torch.from_numpy(lab["x"]).to(dev)
    i = torch.randint(0, len(x), (P,), device=dev, generator=gen)
    rho = torch.exp(x[i, 0]).clamp(max=1.2)
    return rho, EPS * torch.exp(x[i, 1])


def pair_clouds(P, rho, dist, gen, dev):
    s = rho * dist
    a = sy.square_cloud(P, N_PTS, s, gen, dev)
    b = sy.square_cloud(P, N_PTS, s, gen, dev)
    b = b + torch.stack([dist, torch.zeros_like(dist)], 1)[:, None]
    ph = torch.rand(P, device=dev, generator=gen, dtype=torch.float64) * 6.2831853
    R = sy.rot(ph)
    return a @ R.transpose(1, 2), b @ R.transpose(1, 2), s


def sweep_rho(name, kern, head, lab, gen, dev, res):
    kernels.current = kern
    x1 = lab["x"][:, 1]
    dists = [float(EPS * np.exp(np.quantile(x1, q))) for q in (0.2, 0.5, 0.8)]
    rhos = np.logspace(np.log10(0.05), np.log10(1.0), 10)
    out = {"dists": dists, "rhos": rhos.tolist(), "truth": [], "taylor": [], "est": []}
    for d in dists:
        tr, ta, es = [], [], []
        for r in rhos:
            P = 300
            dist = torch.full((P,), d, dtype=torch.float64, device=dev)
            rho = torch.full((P,), float(r), dtype=torch.float64, device=dev)
            a, b, s = pair_clouds(P, rho, dist, gen, dev)
            tr.append(float(sy.label(kern, a, b).mean()))
            ta.append(float(sy.taylor_label(kern, a, b).mean()))
            es.append(float(pred(head, sy.feats(a, b, s, s, N_PTS)).mean()))
        out["truth"].append(tr)
        out["taylor"].append(ta)
        out["est"].append(es)
    lr = np.log(rhos)
    sel = rhos <= 0.4
    for k in ("truth", "taylor", "est"):
        out[k + "_slope"] = [fit_slope(lr[sel], np.array(v)[sel]) for v in out[k]]
    res["rho_sweep"][name] = out


def partial_dep(name, kern, head, lab, gen, dev, res):
    d = float(EPS * np.exp(np.median(lab["x"][:, 1])))
    out = {}
    for rr in (0.1, 0.2, 0.4, 0.8):
        P = 500
        dist = torch.full((P,), d, dtype=torch.float64, device=dev)
        rho = torch.full((P,), rr, dtype=torch.float64, device=dev)
        a, b, s = pair_clouds(P, rho, dist, gen, dev)
        x = sy.feats(a, b, s, s, N_PTS)
        h = 0.1
        sl = {}
        for tag, cols, mult in (("size_only", [0, 3], [1, 1]), ("quad_only", [4, 5], [2, 2]), ("both", [0, 3, 4, 5], [1, 1, 2, 2])):
            xp, xm = x.clone(), x.clone()
            for c, m in zip(cols, mult):
                xp[:, c] += m * h
                xm[:, c] -= m * h
            sl[tag] = float((pred(head, xp) - pred(head, xm)).mean() / (2 * h))
        out[str(rr)] = sl
    res["partial_dep"][name] = out


def aniso(name, kern, head, lab, gen, dev, res):
    kernels.current = kern
    d = float(EPS * np.exp(np.median(lab["x"][:, 1])))
    rho = 0.3
    R = rho * d / 2
    P = 20
    ph = torch.rand(P, device=dev, generator=gen, dtype=torch.float64) * 6.2831853
    Rm = sy.rot(ph)
    off = torch.tensor([d, 0.0], dtype=torch.float64, device=dev)
    out = {"dist": d, "rho": rho}
    for scen in ("tinyA", "both"):
        for kappa, along in ((1, True), (2, True), (4, True), (2, False), (4, False)):
            b = sy.ellipse_cloud(P, R, kappa, along, gen, dev)
            if scen == "tinyA":
                a = sy.ellipse_cloud(P, R, 1, True, gen, dev, tiny=0.03)
                sa = torch.full((P,), 0.06 * R, dtype=torch.float64, device=dev)
            else:
                a = sy.ellipse_cloud(P, R, kappa, along, gen, dev)
                sa = torch.full((P,), 2 * R, dtype=torch.float64, device=dev)
            b = b + off
            a, b = a @ Rm.transpose(1, 2), b @ Rm.transpose(1, 2)
            sb = torch.full((P,), 2 * R, dtype=torch.float64, device=dev)
            tr, ta = sy.label(kern, a, b), sy.taylor_label(kern, a, b)
            x = sy.feats(a, b, sa, sb, a.shape[1])
            es = pred(head, x)
            out[f"{scen}_k{kappa}_{'along' if along else 'perp'}"] = {
                "truth": float(tr.mean()), "truth_std": float(tr.std()), "taylor": float(ta.mean()), "est": float(es.mean()), "est_std": float(es.std()),
                "feat_qrr": float(x[:, 4].mean()), "feat_qtt": float(x[:, 5].mean())}
    res["aniso"][name] = out


def label_stats(name, lab, res):
    x, y = lab["x"], lab["y"]
    c = x[:, 4] - x[:, 5]
    qm = (x[:, 4] + x[:, 5]) / 2
    A = np.stack([np.ones(len(x)), x[:, 0], x[:, 1], x[:, 2], qm, c, x[:, 6]], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    tab = {}
    rb = np.quantile(x[:, 0], [0, 1 / 3, 2 / 3, 1])
    cb = [-9, -0.3, 0.3, 9]
    for i in range(3):
        for j in range(3):
            m = (x[:, 0] >= rb[i]) & (x[:, 0] <= rb[i + 1]) & (c >= cb[j]) & (c < cb[j + 1])
            tab[f"rho{i}_c{j}"] = {"n": int(m.sum()), "mean_y": float(y[m].mean()) if m.sum() else None}
    res["label_stats"][name] = {"ols_names": ["1", "log_rho", "log_dist", "log_count", "meanlogq", "contrast", "corr"], "ols": coef.tolist(), "r2": float(r2),
                                "rho_edges": rb.tolist(), "table": tab, "y_mean": float(y.mean()), "y_std": float(y.std()),
                                "x_min": x.min(0).tolist(), "x_max": x.max(0).tolist(), "x_q05": np.quantile(x, 0.05, 0).tolist(), "x_q95": np.quantile(x, 0.95, 0).tolist()}


def dist_sweep(name, kern, head_own, heads, lab, gen, dev, res):
    kernels.current = kern
    x1 = lab["x"][:, 1]
    dists = EPS * np.exp(np.linspace(np.quantile(x1, 0.02), np.quantile(x1, 0.98), 12))
    out = {"dists": dists.tolist(), "truth": [], "taylor": [], "est": {k: [] for k in heads}}
    for d in dists:
        P = 300
        dist = torch.full((P,), float(d), dtype=torch.float64, device=dev)
        rho = torch.full((P,), 0.2, dtype=torch.float64, device=dev)
        a, b, s = pair_clouds(P, rho, dist, gen, dev)
        out["truth"].append(float(sy.label(kern, a, b).mean()))
        out["taylor"].append(float(sy.taylor_label(kern, a, b).mean()))
        x = sy.feats(a, b, s, s, N_PTS)
        for k, h in heads.items():
            out["est"][k].append(float(pred(h, x).mean()))
    res["dist_sweep"][name] = out


def symmetries(name, kern, head, lab, gen, dev, res):
    kernels.current = kern
    P = 1500
    rho, dist = joint_sample(lab, P, gen, dev)
    a, b, s = pair_clouds(P, rho, dist, gen, dev)
    x = sy.feats(a, b, s, s, N_PTS)
    p0 = pred(head, x)
    out = {"n": P, "pred_std": float(p0.std())}
    xs = sy.feats(b, a, s, s, N_PTS)
    out["exchange_max"] = float((pred(head, xs) - p0).abs().max())
    ang = torch.rand(P, device=dev, generator=gen, dtype=torch.float64) * 6.2831853
    Rm = sy.rot(ang)
    pr = pred(head, sy.feats(a @ Rm.transpose(1, 2), b @ Rm.transpose(1, 2), s, s, N_PTS))
    out["rotation_mean"], out["rotation_max"] = float((pr - p0).abs().mean()), float((pr - p0).abs().max())
    t0 = sy.label(kern, a, b)
    out["rotation_truth_mean"] = float((sy.label(kern, a @ Rm.transpose(1, 2), b @ Rm.transpose(1, 2)) - t0).abs().mean())
    out["pred_vs_truth_rmse"] = float((p0 - t0).pow(2).mean().sqrt())
    sc = {}
    for f in (0.25, 0.5, 2.0, 4.0):
        if name == "yukawa30":
            k2 = kernels.yukawa(0.5 * f, 30.0 * f)
        elif name == "inv_distance":
            k2 = kernels.inv_distance(0.5 * f)
        elif name == "analytic":
            k2 = kernels.analytic(0.5 * f)
        else:
            k2 = ik.get(name, 0.5 * f)
        kernels.current = k2
        a2, b2, s2 = a * f, b * f, s * f
        t2 = sy.label(k2, a2, b2)
        x2 = sy.feats(a2, b2, s2, s2, N_PTS)
        p2 = pred(head, x2)
        xc = x2.clone()
        xc[:, 1] -= float(np.log(f))
        sc[str(f)] = {"truth_shift_mean": float((t2 - t0).mean()), "truth_shift_absmean": float((t2 - t0).abs().mean()), "pred_shift_mean": float((p2 - p0).mean()),
                      "pred_shift_absmean": float((p2 - p0).abs().mean()), "pred_err_rmse": float((p2 - t2).pow(2).mean().sqrt()),
                      "corrected_shift_absmean": float((pred(head, xc) - p0).abs().mean()), "x1_out_of_support": float(((x2[:, 1] < float(lab["x"][:, 1].min())) | (x2[:, 1] > float(lab["x"][:, 1].max()))).double().mean())}
    kernels.current = kern
    out["scale"] = sc
    res["symmetry"][name] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/interp_stage1.json")
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.smoke else "cuda")
    gen = torch.Generator(device=dev).manual_seed(4738)
    res = {k: {} for k in ("rho_sweep", "partial_dep", "aniso", "label_stats", "dist_sweep", "symmetry")}
    heads, labs, kerns = {}, {}, {}
    for n in a.kernels.split(","):
        heads[n] = est_train.load(f"checkpoints/interp_est_{n}.pt", dev)
        d = np.load(f"results/interp_labels_{n}.npz")
        labs[n] = {"x": d["x"], "y": d["y"]}
        kerns[n] = ik.get(n)
    for n in heads:
        print(n, flush=True)
        label_stats(n, labs[n], res)
        sweep_rho(n, kerns[n], heads[n], labs[n], gen, dev, res)
        partial_dep(n, kerns[n], heads[n], labs[n], gen, dev, res)
        aniso(n, kerns[n], heads[n], labs[n], gen, dev, res)
        symmetries(n, kerns[n], heads[n], labs[n], gen, dev, res)
        json.dump(res, open(a.out, "w"))
    for n in ("analytic", "yukawa30"):
        if n in heads:
            hs = {"own": heads[n]}
            if n == "yukawa30" and "analytic" in heads:
                hs["analytic_est"] = heads["analytic"]
            if n == "analytic" and "yukawa30" in heads:
                hs["yukawa30_est"] = heads["yukawa30"]
            dist_sweep(n, kerns[n], heads[n], hs, labs[n], gen, dev, res)
    json.dump(res, open(a.out, "w"))
