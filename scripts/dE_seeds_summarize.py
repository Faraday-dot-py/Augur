"""Summarise results/dE_seeds/*.npz (numpy only).

Per regime and arm: median |dE|/(K0+|U0|) at ticks 100 and 1000 with 95% bootstrap CI over scenes, median outside-grid fraction. Per pairing (B over A):
ratio of medians with a paired bootstrap CI (same resampled scenes in both arms, 4000 resamples, seed 4738), median of per-scene ratios, fraction of scenes B < A.
Primary G/E pairing = det arms if the E det reruns are bitwise identical, else nd rep 1.

Usage: python scripts/dE_seeds_summarize.py results/dE_seeds [out.txt]
"""
import glob
import os
import sys

import numpy as np

GATE = 1.15
NBOOT = 4000


def load(d):
    return {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(f"{d}/*.npz"))}


def dE(a, reg, t):
    i = int(np.searchsorted(a["ticks"], t))
    return np.abs(a[f"{reg}_E"][i] - a[f"{reg}_E"][0]) / (a[f"{reg}_K"][0] + np.abs(a[f"{reg}_U"][0]))


def at(a, reg, k, t):
    return a[f"{reg}_{k}"][int(np.searchsorted(a["ticks"], t))]


def boot_med(x, rng):
    idx = rng.integers(0, len(x), (NBOOT, len(x)))
    m = np.median(x[idx], 1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def pair(a, b, rng):
    n = len(a)
    idx = rng.integers(0, n, (NBOOT, n))
    r = np.median(b[idx], 1) / np.median(a[idx], 1)
    ok = (a > 0) & (b > 0)
    return np.median(b) / np.median(a), np.percentile(r, 2.5), np.percentile(r, 97.5), float(np.median(b[ok] / a[ok])), float(np.mean(b < a))


def verdict(lo, hi):
    return "PASS" if hi <= GATE else ("KILL" if lo > GATE else "INCONCLUSIVE")


def main():
    d = sys.argv[1]
    arms = load(d)
    rng = np.random.default_rng(4738)
    lines = []
    P = lines.append
    regs = sorted({k.split("_")[0] for a in arms.values() for k in a if k.endswith("_E") and k not in ("ticks",)})
    P(f"arms: {sorted(arms)}; scenes per arm: {len(next(iter(arms.values()))['seeds'])}")
    for reg in regs:
        P(f"\n== {reg} ==")
        P("| arm | dE@100 med [95% CI] | dE@1e3 med [95% CI] | outgrid@100 | outgrid@1e3 | bound@1e3 | vir@1e3 | nonfinite scenes |")
        P("|---|---|---|---|---|---|---|---|")
        for nm, a in arms.items():
            c = []
            for t in (100, 1000):
                x = dE(a, reg, t)
                lo, hi = boot_med(x, rng)
                c.append(f"{np.median(x):.4g} [{lo:.4g}, {hi:.4g}]")
            nf = int((at(a, reg, "finite", 1000) < 0.5).sum())
            P(f"| {nm} | {c[0]} | {c[1]} | {np.median(at(a, reg, 'outgrid', 100)):.3f} | {np.median(at(a, reg, 'outgrid', 1000)):.3f} | "
              f"{np.median(at(a, reg, 'bound', 1000)):.3f} | {np.median(at(a, reg, 'vir', 1000)):.3g} | {nf} |")
        det_same = None
        if "E_det1" in arms and "E_det2" in arms:
            det_same = bool(np.array_equal(arms["E_det1"][f"{reg}_E"], arms["E_det2"][f"{reg}_E"]))
            gs = bool(np.array_equal(arms["G_det1"][f"{reg}_E"], arms["G_det2"][f"{reg}_E"])) if "G_det1" in arms and "G_det2" in arms else None
            P(f"\ndeterminism: E det1 == det2 bitwise on E(t): {det_same}; G det1 == det2: {gs}")
        pairs = []
        for k in "123":
            pairs.append((f"G_det{k}/E_det{k}", f"E_det{k}", f"G_det{k}", "ghost vs E, deterministic"))
            pairs.append((f"G_nd{k}/E_nd{k}", f"E_nd{k}", f"G_nd{k}", "ghost vs E, nondeterministic"))
        pairs += [("E_nd2/E_nd1", "E_nd1", "E_nd2", "noise floor E-vs-E"), ("E_nd3/E_nd1", "E_nd1", "E_nd3", "noise floor E-vs-E"),
                  ("E_det2/E_det1", "E_det1", "E_det2", "noise floor E-vs-E det"), ("G_nd2/G_nd1", "G_nd1", "G_nd2", "noise floor G-vs-G"),
                  ("G_nd3/G_nd1", "G_nd1", "G_nd3", "noise floor G-vs-G"), ("E_det1/E_nd1", "E_nd1", "E_det1", "det vs nd, same model"),
                  ("G_det1/G_nd1", "G_nd1", "G_det1", "det vs nd, same model")]
        pairs += [("B_nd1/E_nd1", "E_nd1", "B_nd1", "penalty arm vs E"), ("C_nd1/E_nd1", "E_nd1", "C_nd1", "control vs E"), ("B_nd1/C_nd1", "C_nd1", "B_nd1", "penalty vs control")]
        P("\n| pairing B/A | what | t | ratio of medians [paired 95% CI] | median of per-scene ratios | frac scenes B<A | vs 1.15 gate |")
        P("|---|---|---|---|---|---|---|")
        for nm, an, bn, what in pairs:
            if an not in arms or bn not in arms:
                continue
            for t in (100, 1000):
                r, lo, hi, pm, fr = pair(dE(arms[an], reg, t), dE(arms[bn], reg, t), rng)
                v = verdict(lo, hi) if what.startswith(("ghost", "penalty", "control")) else ""
                P(f"| {nm} | {what} | {t} | {r:.3f} [{lo:.3f}, {hi:.3f}] | {pm:.3f} | {fr:.2f} | {v} |")
        pool = [n for n in arms if n.startswith("G_nd")], [n for n in arms if n.startswith("E_nd")]
        if pool[0] and pool[1]:
            for t in (100, 1000):
                ga = np.mean([dE(arms[n], reg, t) for n in pool[0]], 0)
                ea = np.mean([dE(arms[n], reg, t) for n in pool[1]], 0)
                r, lo, hi, pm, fr = pair(ea, ga, rng)
                P(f"| mean of {len(pool[0])} nd reps G / mean of {len(pool[1])} nd reps E | pooled nondeterministic | {t} | {r:.3f} [{lo:.3f}, {hi:.3f}] | {pm:.3f} | {fr:.2f} | {verdict(lo, hi)} |")
    out = "\n".join(lines)
    print(out)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out + "\n")


if __name__ == "__main__":
    main()
