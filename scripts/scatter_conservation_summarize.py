"""Summaries and diagnostic grids from results/conservation/<regime>.npz (scripts/scatter_conservation.py).

Usage: python scripts/scatter_conservation_summarize.py --dir results/conservation --out-json results/conservation/summary.json --fig-dir videos
Prints markdown tables; writes videos/scatter_conservation_<regime>_ts.png (time series) and _snap.png (positions, seed-4738 scene 0).
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETS = (("m", "model", "C3"), ("t4", "truth 4 sub", "k"), ("t1", "truth 1 sub", "C0"), ("t16", "truth 16 sub", "0.5"))
CHECK = (100, 1000, 10000)
C = 10.0


def derived(d, s, n):
    K0, U0 = d[f"{s}_K"][0], d[f"{s}_U"][0]
    r0 = d["r0"]
    esc = K0 + np.abs(U0)
    lsc = r0 * np.sqrt(2 * n * np.maximum(K0, 1e-12))
    return {"dE": (d[f"{s}_E"] - d[f"{s}_E"][0]) / esc, "dL": (d[f"{s}_L"] - d[f"{s}_L"][0]) / lsc,
            "com": np.hypot(d[f"{s}_comx"] - d[f"{s}_comx"][0], d[f"{s}_comy"] - d[f"{s}_comy"][0]), "P": np.abs(d[f"{s}_P"] - d[f"{s}_P"][0]),
            "vir": d[f"{s}_vir"], "bound": d[f"{s}_bound"], "ej": d[f"{s}_ejected"] / n, "out": d[f"{s}_outgrid"],
            "rmin": d[f"{s}_rmin"], "close": d[f"{s}_close"], "vmax": d[f"{s}_vmax"], "f90": d[f"{s}_frac90"], "rhalf": d[f"{s}_rhalf"],
            "rmax": d[f"{s}_rmax"], "amax": d[f"{s}_amax"]}


def at(arr, ticks, t):
    return arr[int(np.searchsorted(ticks, t))]


def med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/conservation")
    ap.add_argument("--out-json", default="results/conservation/summary.json")
    ap.add_argument("--fig-dir", default="videos")
    args = ap.parse_args()
    summary = {}
    for f in sorted(glob.glob(f"{args.dir}/*.npz")):
        name = os.path.basename(f)[:-4]
        d = dict(np.load(f))
        ticks = d["ticks"]
        n = d["m_snaps"].shape[2]
        tmax = int(ticks[-1])
        checks = [t for t in CHECK if t <= tmax] or [tmax]
        reg = {"n": n, "B": int(len(d["seeds"])), "tmax": tmax}
        fig, ax = plt.subplots(2, 5, figsize=(22, 8))
        for s, label, col in SETS:
            if f"{s}_E" not in d:
                continue
            q = derived(d, s, n)
            reg[s] = {"first_nonfinite": d[f"{s}_first_nonfinite"].tolist(), "vmax_all_over_c": float(np.nanmax(d[f"{s}_vmax_all"]) / C),
                      "first_ge_c": d[f"{s}_first_ge_c"].tolist(), "rmin_all_min": float(np.min(d[f"{s}_rmin_all"])), "amax_all_max": float(np.max(d[f"{s}_amax_all"]))}
            for t in checks:
                reg[s][str(t)] = {k: {"median": med(at(v, ticks, t)), "p84": float(np.nanpercentile(at(v, ticks, t), 84)), "p16": float(np.nanpercentile(at(v, ticks, t), 16))}
                                  for k, v in q.items()}
                reg[s][str(t)]["abs_dE_median"] = med(np.abs(at(q["dE"], ticks, t)))
                reg[s][str(t)]["abs_dL_median"] = med(np.abs(at(q["dL"], ticks, t)))
            panels = (("dE", "dE/(K0+|U0|) (signed)", "linear"), ("dL", "dL/Lscale", "linear"), ("com", "COM drift", "log"), ("vir", "virial 2K/|U|", "log"),
                      ("bound", "bound fraction", "linear"), ("ej", "ejected / N", "linear"), ("out", "frac outside grid", "linear"),
                      ("rhalf", "median radius", "log"), ("rmin", "min pair distance", "log"), ("amax", "max |dv|/dt", "log"))
            for a, (k, ttl, sc) in zip(ax.ravel(), panels):
                v = q[k]
                lo, mid, hi = np.nanpercentile(v, 16, axis=1), np.nanmedian(v, axis=1), np.nanpercentile(v, 84, axis=1)
                a.plot(ticks + 1, mid, color=col, label=label, lw=1.5)
                a.fill_between(ticks + 1, lo, hi, color=col, alpha=0.15)
                a.set_xscale("log")
                if sc == "log":
                    a.set_yscale("log")
                a.set_title(ttl, fontsize=9)
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f"{name}: N={n}, {reg['B']} scenes, median and 16-84% over scenes, x = tick+1")
        plt.tight_layout()
        os.makedirs(args.fig_dir, exist_ok=True)
        plt.savefig(f"{args.fig_dir}/scatter_conservation_{name}_ts.png", dpi=70)
        plt.close(fig)
        ms, ts = d["m_snaps"], d["t4_snaps"]
        pick = [t for t in (0, 50, 100, 300, 1000, 3000, 10000) if t <= tmax and t // 10 < ms.shape[0]]
        fig, ax = plt.subplots(2, len(pick), figsize=(3.1 * len(pick), 6.4), squeeze=False)
        ext = 1.2 * np.abs(ts[:, 0]).max()
        for j, t in enumerate(pick):
            for i, (arr, lab) in enumerate(((ts, "truth"), (ms, "model"))):
                p = arr[t // 10, 0]
                ax[i, j].plot(p[:, 0], p[:, 1], ".", ms=2 if n > 50 else 6)
                ax[i, j].set_xlim(-ext, ext)
                ax[i, j].set_ylim(-ext, ext)
                ax[i, j].axvline(-32, color="r", lw=0.4)
                ax[i, j].axvline(32, color="r", lw=0.4)
                ax[i, j].axhline(-32, color="r", lw=0.4)
                ax[i, j].axhline(32, color="r", lw=0.4)
                ax[i, j].set_aspect("equal")
                ax[i, j].set_title(f"{lab} t={t}", fontsize=8)
        fig.suptitle(f"{name} seed 4738 scene 0 (red box = model grid)")
        plt.tight_layout()
        plt.savefig(f"{args.fig_dir}/scatter_conservation_{name}_snap.png", dpi=60)
        plt.close(fig)
        summary[name] = reg
        print(f"\n### {name} (N={n}, B={reg['B']})")
        print("| set | t | |dE|/scale med | dE signed med | |dL|/Lscale med | |P| med | COM drift med | virial med | bound med | ejected/N | outside grid | rmin_all | vmax/c |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s, label, _ in SETS:
            if s not in reg:
                continue
            for t in checks:
                r = reg[s][str(t)]
                print(f"| {s} | {t} | {r['abs_dE_median']:.3g} | {r['dE']['median']:.3g} | {r['abs_dL_median']:.3g} | {r['P']['median']:.2g} | {r['com']['median']:.3g} | "
                      f"{r['vir']['median']:.3g} | {r['bound']['median']:.3g} | {r['ej']['median']:.3g} | {r['out']['median']:.3g} | {reg[s]['rmin_all_min']:.3g} | {reg[s]['vmax_all_over_c']:.3g} |")
        print(f"first non-finite (model): {reg['m']['first_nonfinite']}  first |v|>=c: {reg['m']['first_ge_c']}")
    json.dump(summary, open(args.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
