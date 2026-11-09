"""Rule-free force recovery, step 2: menu-free expression discovery on a radial force curve (docs/debugging/rule-free-force-recovery.md).
Library of 18 expressions (6 bases x 3 wrappers) fitted by differential evolution (seed 4738) in z = asinh(F/c) space, c = 1e-3 max|F| of the curve being fitted; selection by
BIC with a 0.01 residual floor, ties (BIC within 2) -> fewest parameters, then lowest rss, then name. Pre-registered in docs/debugging/experiment-log.md (2026-11-08).
Usage: PYTHONPATH=. python scripts/rule_free_force_discovery.py exact --out results/rule_free_force_exact.json
       PYTHONPATH=. python scripts/rule_free_force_discovery.py model --curves results/rule_free_force_curves --out results/rule_free_force_model.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from scipy.optimize import differential_evolution

from scripts import scatter_multi as sm

LAWS = ("pow1", "pow1.5", "pow2", "pow3", "yuk8", "yuk32", "yuk128", "lj")
D_FIT = np.geomspace(0.25, 14.0, 56)
D_HELD = np.sqrt(D_FIT[:-1] * D_FIT[1:])
SEED = 4738
FLOOR = 0.01
TIE = 2.0

AMP = ("amp", -8.0, 8.0)
EXPO = ("expo", 0.05, 16.0)
SOFT = ("soft", 0.02, 4.0)
LOGL = ("logL", 0.0, 3.7)
BASES = {
    "power": [AMP, EXPO],
    "soft-power": [AMP, EXPO, SOFT],
    "exponential": [AMP, LOGL],
    "power-exponential": [AMP, EXPO, LOGL],
    "soft-power-exponential": [AMP, EXPO, SOFT, LOGL],
    "two-power": [AMP, EXPO, AMP, EXPO],
}
WRAPS = {
    "": [],
    "+sat": [("logC", -0.5, 2.5)],
    "+clamp+sat": [("clamp", 0.1, 4.0), ("logC", -0.5, 2.5)],
}


def amp(v):
    return np.sign(v) * (10.0 ** np.abs(v) - 1.0)


def evaluate(base, wrap, x, d):
    nb = len(BASES[base])
    p = x[:nb]
    w = x[nb:]
    if wrap == "+clamp+sat":
        d = np.maximum(d, w[0])
    if base == "power":
        f = amp(p[0]) * d ** -p[1]
    elif base == "soft-power":
        f = amp(p[0]) * d * (d ** 2 + p[2] ** 2) ** (-(p[1] + 1) / 2)
    elif base == "exponential":
        f = amp(p[0]) * np.exp(-d / 10.0 ** p[1])
    elif base == "power-exponential":
        f = amp(p[0]) * d ** -p[1] * np.exp(-d / 10.0 ** p[2])
    elif base == "soft-power-exponential":
        f = amp(p[0]) * d * (d ** 2 + p[2] ** 2) ** (-(p[1] + 1) / 2) * np.exp(-d / 10.0 ** p[3])
    else:
        f = amp(p[0]) * d ** -p[1] + amp(p[2]) * d ** -p[3]
    if wrap:
        C = 10.0 ** w[-1]
        f = C * np.tanh(f / C)
    return f


def describe(base, wrap, x):
    nb = len(BASES[base])
    p = x[:nb]
    w = x[nb:]
    L = lambda v: f"{10.0 ** v:.4g}"
    if base == "power":
        s = f"{amp(p[0]):.4g} d^-{p[1]:.4g}"
    elif base == "soft-power":
        s = f"{amp(p[0]):.4g} d (d^2+{p[2]:.3g}^2)^-({p[1]:.4g}+1)/2"
    elif base == "exponential":
        s = f"{amp(p[0]):.4g} exp(-d/{L(p[1])})"
    elif base == "power-exponential":
        s = f"{amp(p[0]):.4g} d^-{p[1]:.4g} exp(-d/{L(p[2])})"
    elif base == "soft-power-exponential":
        s = f"{amp(p[0]):.4g} d (d^2+{p[2]:.3g}^2)^-({p[1]:.4g}+1)/2 exp(-d/{L(p[3])})"
    else:
        s = f"{amp(p[0]):.4g} d^-{p[1]:.4g} + {amp(p[2]):.4g} d^-{p[3]:.4g}"
    if wrap == "+clamp+sat":
        s = f"[d -> max(d,{w[0]:.3g})] " + s
    if wrap:
        s = f"{L(w[-1])} tanh(F/{L(w[-1])}), F = " + s
    return s


def zt(f, c):
    return np.arcsinh(f / c)


def fit_one(args):
    base, wrap, d, y, c = args
    bounds = [(lo, hi) for _, lo, hi in BASES[base] + WRAPS[wrap]]
    zy = zt(y, c)

    def obj(x):
        with np.errstate(all="ignore"):
            f = evaluate(base, wrap, x, d)
        if not np.all(np.isfinite(f)):
            return 1e6
        return float(np.sum((zt(f, c) - zy) ** 2))

    r = differential_evolution(obj, bounds, seed=SEED, popsize=25, maxiter=600, tol=1e-12, mutation=(0.4, 1.0), recombination=0.8, polish=True, init="sobol")
    x = r.x
    k = len(bounds)
    return {"name": base + wrap, "base": base, "wrap": wrap, "k": k, "rss": float(r.fun), "x": x.tolist(), "expr": describe(base, wrap, x)}


def select(cands, n):
    for c in cands:
        c["bic"] = n * np.log(max(c["rss"] / n, FLOOR ** 2)) + c["k"] * np.log(n)
    best = min(c["bic"] for c in cands)
    tied = sorted([c for c in cands if c["bic"] <= best + TIE], key=lambda c: (c["k"], c["rss"], c["name"]))
    return tied[0], [c["name"] for c in tied]


def held_error(c, y_true_fit, d_held, y_true_held):
    cs = 1e-3 * np.max(np.abs(y_true_fit))
    with np.errstate(all="ignore"):
        f = evaluate(c["base"], c["wrap"], np.array(c["x"]), d_held)
    return float(np.sqrt(np.mean((zt(f, cs) - zt(y_true_held, cs)) ** 2)))


def exact_curve(law, d):
    slots = sm.slots_for("single", law, 1.0)
    par = torch.tensor(sm.params_of(slots), dtype=torch.float64)[None]
    pos = torch.zeros(len(d), 2, 2, dtype=torch.float64)
    pos[:, 1, 0] = torch.tensor(d)
    typ = torch.zeros(len(d), 2)
    mask = torch.ones(len(d), 2, dtype=torch.float64)
    a = sm.exact_acc(pos, typ, mask, par.expand(len(d), -1, -1).contiguous())
    return a[:, 0, 0].numpy()


def discover(pool, jobs):
    tasks = [(base, wrap, d, y, 1e-3 * np.max(np.abs(y))) for (_, d, y) in jobs for base in BASES for wrap in WRAPS]
    fits = pool.map(fit_one, tasks, chunksize=1)
    out = []
    nc = len(BASES) * len(WRAPS)
    for i, (key, d, y) in enumerate(jobs):
        cands = fits[i * nc:(i + 1) * nc]
        sel, tied = select(cands, len(d))
        out.append((key, cands, sel, tied))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("exact", "model"))
    ap.add_argument("--curves", default="results/rule_free_force_curves")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    truth = {law: (exact_curve(law, D_FIT), exact_curve(law, D_HELD)) for law in LAWS}
    jobs = []
    if a.mode == "exact":
        for law in LAWS:
            jobs.append((f"exact|{law}", D_FIT, truth[law][0]))
    else:
        for ck in ("mixed_s4738", "mixed_s4739"):
            for law in LAWS:
                z = np.load(os.path.join(a.curves, f"{ck}_{law}.npz"))
                for s in (4738, 4739, 4740):
                    jobs.append((f"{ck}|{law}|{s}", D_FIT, z[f"fit_{s}_med"]))
    with Pool(a.workers) as pool:
        res = discover(pool, jobs)
    out = {}
    for key, cands, sel, tied in res:
        law = key.split("|")[1]
        tf, th = truth[law]
        sel["E_held"] = held_error(sel, tf, D_HELD, th)
        for c in cands:
            c["E_held"] = held_error(c, tf, D_HELD, th)
        out[key] = {"selected": sel, "tied": tied, "candidates": cands}
        print(key, "->", sel["name"], "k", sel["k"], "rms_z %.4f" % np.sqrt(sel["rss"] / len(D_FIT)), "E_held %.4f" % sel["E_held"], "tied", tied, "|", sel["expr"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
