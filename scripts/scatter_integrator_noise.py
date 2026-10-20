"""Does a better integrator help once the force has E-sized error? (docs/debugging/learned-integrator-research.md; CPU, fp64, no model.)

  PYTHONPATH=. python scripts/scatter_integrator_noise.py --out results/integrator_noise.json
Part A: expB (24 scenes, seed 9300) and cluster (12 scenes, seed 9200), 100-tick free rollouts from truth initial states with exact force plus iid relative force
noise sigma (as scatter_rootcause.noisy_fa), KDK with M = 1 (the model's integrator) / 2 / 4 substeps per tick, err@5/20/50/100 vs the 4-substep truth.
Part B: nr100 scenes (conservation-harness seeds 9100/9200/9300/4738), 1000 ticks, same arms, median |dE|/(K0+|U0|) at ticks 100 and 1000.
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_conservation as sc_cons
from scripts import scatter_substep_theory as th

DT, EPS = th.DT, th.EPS
SIGMAS = (0.0, 0.02, 0.08)
MS = (1, 2, 4)
KS = (5, 20, 50, 100)


COHERENT = False


def noisy_acc(p, m, sigma, xi):
    a = th.acc(p, m)
    if sigma > 0:
        a = a * (1 + sigma * xi)
        a = a - a.mean(0, keepdim=True)
    return a


def roll(p, v, m, M, sigma, gen, steps):
    h = DT / M
    n = len(p)
    xis = [torch.randn((n, 2), generator=gen, dtype=p.dtype) for _ in range(steps + 1)]
    a = noisy_acc(p, m, sigma, xis[0])
    P, V = [p], [v]
    for k in range(steps):
        for j in range(M):
            s = (j + 1) / M
            xi = (1 - s) * xis[k] + s * xis[k + 1] if COHERENT else torch.randn((n, 2), generator=gen, dtype=p.dtype)
            v = v + 0.5 * h * a
            p = p + h * v
            a = noisy_acc(p, m, sigma, xi if not COHERENT or j < M - 1 else xis[k + 1])
            v = v + 0.5 * h * a
        P.append(p)
        V.append(v)
    return torch.stack(P), torch.stack(V)


def part_a(log):
    out = {}
    for fam, num, seed in (("expB", 24, 9300), ("cluster", 12, 9200)):
        scenes = th.make_scenes(fam, num, seed)
        res = {}
        for sigma in SIGMAS:
            for M in MS:
                gen = torch.Generator().manual_seed(4738)
                errs = []
                for sc in scenes:
                    P, _ = roll(sc["P"][0], sc["V"][0], sc["m"], M, sigma, gen, th.TR)
                    errs.append((P - sc["P"][:th.TR + 1]).norm(dim=-1).mean(1).numpy())
                c = np.mean(errs, 0)
                res[f"s{sigma}_M{M}"] = {str(k): float(c[k]) for k in KS}
                log(f"{fam} sigma {sigma} M {M}: " + " ".join(f"{k}:{c[k]:.5f}" for k in KS))
        out[fam] = res
    return out


def energy(p, v, m):
    d = p[None] - p[:, None]
    r = torch.sqrt((d ** 2).sum(-1) + EPS ** 2)
    iu = torch.triu_indices(len(p), len(p), 1)
    return 0.5 * (m * (v ** 2).sum(-1)).sum() - (m[iu[0]] * m[iu[1]] / r[iu[0], iu[1]]).sum()


def part_b(log, steps):
    ps, vs, seeds = sc_cons.make_scenes("nr", 100, 1)
    res = {}
    for sigma in SIGMAS:
        for M in MS:
            gen = torch.Generator().manual_seed(4738)
            d100, d1000 = [], []
            for i in range(len(ps)):
                p, v = torch.tensor(ps[i]), torch.tensor(vs[i])
                m = torch.ones(len(p))
                P, V = roll(p, v, m, M, sigma, gen, steps)
                e0 = energy(P[0], V[0], m)
                k0 = 0.5 * (V[0] ** 2).sum()
                scale = k0 + (k0 - e0).abs()
                for t, lst in ((100, d100), (steps, d1000)):
                    lst.append(float((energy(P[t], V[t], m) - e0) / scale))
            res[f"s{sigma}_M{M}"] = {"dE@100": d100, "dE@1000": d1000,
                                      "median_abs@100": float(np.median(np.abs(d100))), "median_abs@1000": float(np.median(np.abs(d1000)))}
            log(f"nr100 sigma {sigma} M {M}: |dE| median @100 {res[f's{sigma}_M{M}']['median_abs@100']:.5g} @1000 {res[f's{sigma}_M{M}']['median_abs@1000']:.5g}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/integrator_noise.json")
    ap.add_argument("--steps-b", type=int, default=1000)
    ap.add_argument("--coherent", action="store_true", help="force error smooth in time within a tick (linear interpolation of per-tick noise) instead of iid per evaluation")
    args = ap.parse_args()
    global COHERENT
    COHERENT = args.coherent
    log = lambda s: print(s, flush=True)
    out = {"A": part_a(log), "B": part_b(log, args.steps_b)}
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
