"""Compare model conservation rows between result dirs (npz from scatter_conservation.py), numpy only.
Usage: python scripts/summarize_cons_compare.py <dirA> <dirB> [regimes...]     (A = control, B = variant)
Per regime and dir: median |dE|/(K0+|U0|) at ticks 100/1000/10000, bound fraction, virial, max |v|/c, first non-finite.
If both dirs have the same number of scenes: paired ratio B/A (median of per-scene ratios), ratio of medians with a 95% bootstrap interval
(2000 resamples of scenes, seed 4738), and fraction of scenes where B < A.
"""
import glob
import os
import sys

import numpy as np

C = 10.0


def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def at(a, ticks, t):
    return a[int(np.searchsorted(ticks, t))]


def dE_at(d, t, s="m"):
    ticks = d["ticks"]
    K0, U0 = d[f"{s}_K"][0], d[f"{s}_U"][0]
    return at(np.abs(d[f"{s}_E"] - d[f"{s}_E"][0]) / (K0 + np.abs(U0)), ticks, t)


def row(d, s="m"):
    ticks = d["ticks"]
    out = {}
    for t in (100, 1000, 10000):
        if t <= int(ticks[-1]):
            out[t] = (med(dE_at(d, t, s)), med(at(d[f"{s}_bound"], ticks, t)), med(at(d[f"{s}_vir"], ticks, t)))
    return out, float(np.nanmax(d[f"{s}_vmax_all"]) / C), d[f"{s}_first_nonfinite"].tolist()


def main():
    dirs = sys.argv[1:3]
    regs = sys.argv[3:] or sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{dirs[0]}/*.npz"))
    rng = np.random.default_rng(4738)
    for r in regs:
        ds = {}
        for dd in dirs:
            f = f"{dd}/{r}.npz"
            if not os.path.exists(f):
                continue
            ds[dd] = dict(np.load(f))
            out, vmax, nf = row(ds[dd])
            B = len(ds[dd]["seeds"])
            print(f"{r:6s} {dd:18s} B={B} " + "  ".join(f"t{t}: |dE| {v[0]:.4g} bound {v[1]:.3g} vir {v[2]:.3g}" for t, v in out.items()) + f"  vmax/c {vmax:.3f} nonfinite {sorted(set(nf))}")
        if len(ds) == 2 and len(ds[dirs[0]]["seeds"]) == len(ds[dirs[1]]["seeds"]):
            A, Bd = ds[dirs[0]], ds[dirs[1]]
            for t in (100, 1000):
                a, b = dE_at(A, t), dE_at(Bd, t)
                n = len(a)
                idx = rng.integers(0, n, (2000, n))
                ratios = np.median(b[idx], 1) / np.median(a[idx], 1)
                print(f"   t{t}: median ratio {med(b) / med(a):.3f} (95% boot {np.percentile(ratios, 2.5):.3f}..{np.percentile(ratios, 97.5):.3f}), "
                      f"paired median ratio {med(b / a):.3f}, frac scenes B<A {float(np.mean(b < a)):.2f}")


if __name__ == "__main__":
    main()
