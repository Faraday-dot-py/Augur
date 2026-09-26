"""Helpers for the ALF generalization test (genic): state sets, static accuracy/cost sweeps, estimator calibration, estimator retraining."""
import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import kernels
from scripts import est_train

CAP, THETA_MAX = 8, 1.2
GEO_THETAS = [0.2, 0.25, 0.35, 0.5, 0.7, 1.0]
BH_THETAS = [0.2, 0.25, 0.35, 0.5, 0.7]
TOLS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def load_snap_state(tag, step, dev):
    d = np.load(f"results/rollout_{tag}_snaps.npz")
    i = list(d["steps"]).index(step)
    return torch.from_numpy(d["pos"][i].copy()).to(dev).double()


def make_targets(n, k, dev):
    return torch.randperm(n, generator=torch.Generator().manual_seed(4738))[:k].to(dev)


def sweep(pos, heads, S, tols=TOLS, geo=True, bh=True):
    """Kernel evals per particle and error vs exact (analytic kernel) on targets S for geo dual tree, BH and each estimator head."""
    kernels.current = kernels.analytic()
    N = pos.shape[0]
    a_ex = ao.exact_accel(pos, S)
    tr = ao.Tree(pos, 18)
    fl = dt.Flat(tr)
    out = {"N": N, "geo": [], "bh": [], "est": {}}
    if geo:
        for th in GEO_THETAS:
            a_s, m2l, dirn = dt.dual_accel(tr, fl, th, CAP)
            a = torch.empty_like(a_s)
            a[tr.order] = a_s
            out["geo"].append({"theta": th, "cost": (m2l + dirn) / N, **de.metrics2(a[S], a_ex)})
    if bh:
        for th in BH_THETAS:
            acc, nm, ns = ao.accel_all(tr, tr.inv[S], CAP, theta=th)
            out["bh"].append({"theta": th, "cost": float((nm + ns).mean()), **de.metrics2(acc, a_ex)})
    a_c, _, _ = dt.dual_accel(tr, fl, THETA_MAX, CAP)
    fs = de.node_scale(fl, a_c)
    for name, h in heads.items():
        rows = []
        for tol in tols:
            r, _ = de.run_est(tr, fl, S, a_ex, CAP, h, 1.0, tol, fs, THETA_MAX)
            r["tol"] = tol
            rows.append(r)
        out["est"][name] = rows
    return out


def calib(pos, head, gen, n_pairs=40000, n_big=4000):
    """R2, bias (mean yhat - y, log units), q90 / mean-head coverage on freshly labelled node pairs of this state."""
    kernels.current = kernels.analytic()
    tr = ao.Tree(pos, 18)
    fl = dt.Flat(tr)
    ga, gb = de.collect_pairs(tr, fl, [0.35, 0.7, 1.2], CAP, n_pairs, n_big, gen)
    y, _, _ = de.label(tr, fl, ga, gb, gen)
    x = de.pair_feats(fl, ga, gb)
    keep = torch.isfinite(x).all(1) & torch.isfinite(y)
    x, y = x[keep], y[keep]
    with torch.no_grad():
        m = de.Head(head.est, 0)(x)
        q = de.Head(head.est, 1)(x)
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": int(len(y)), "r2": float(1 - ((y - m) ** 2).sum() / ss), "bias": float((m - y).mean()), "cover_q90": float((y <= q).double().mean()),
            "frac_mean_underestimate": float((y > m).double().mean()), "y_mean": float(y.mean()), "y_std": float(y.std())}


def pair_data(pos, gen, n_pairs=40000, n_big=4000):
    kernels.current = kernels.analytic()
    tr = ao.Tree(pos, 18)
    fl = dt.Flat(tr)
    ga, gb = de.collect_pairs(tr, fl, [0.35, 0.7, 1.2], CAP, n_pairs, n_big, gen)
    y, _, _ = de.label(tr, fl, ga, gb, gen)
    x = de.pair_feats(fl, ga, gb)
    keep = torch.isfinite(x).all(1) & torch.isfinite(y)
    return x[keep].cpu(), y[keep].cpu()


def train_on(data, dev, epochs=40, seed=4738):
    x, y = torch.cat([d[0] for d in data]), torch.cat([d[1] for d in data])
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    nv = len(x) // 10
    est = de.train(x[perm[nv:]], y[perm[nv:]], x[perm[:nv]], y[perm[:nv]], [0.9], dev, epochs, seed)
    return de.Head(est, 0)


def cost_at(rows, key, target):
    """Log-log interpolated cost at error `target` from rows with fields cost and key (sorted by error); None if outside the range."""
    pts = sorted((r[key], r["cost"]) for r in rows)
    e = np.log([p[0] for p in pts])
    c = np.log([p[1] for p in pts])
    t = np.log(target)
    if t < e.min() or t > e.max():
        return None
    return float(np.exp(np.interp(t, e, c)))
