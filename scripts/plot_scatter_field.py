"""Local plots for the scatter-field experiments from downloaded results JSON/NPZ. Usage:
python scripts/plot_scatter_field.py --res ~/polaris-mcp-files/sf/res --out videos"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = "9000"


def load(res, exp):
    out = {}
    for p in sorted(glob.glob(f"{res}/{exp}_*.json")):
        name = os.path.basename(p)[len(exp) + 1:-5]
        if name.endswith("_train"):
            continue
        out[name] = json.load(open(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="videos")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, exp, title in zip(axs, "ABD", ("A: 2 bodies", "B: 10-100 bodies", "D: 2-body orbits d=0.25..16 h")):
        R = load(args.res, exp)
        for name, r in R.items():
            if SEED not in r:
                continue
            ax.plot(range(1, 101), r[SEED]["err"], label=name, lw=2 if name == "baseline" else 1, ls="--" if name == "baseline" else "-")
        if R:
            ref = next(r for r in R.values() if SEED in r)
            ax.plot(range(1, 101), ref[SEED]["const_vel_err"], "k:", label="const vel")
        ax.axvline(30, color="gray", lw=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step (train windows <= 20, data 30)")
        ax.set_ylabel("mean position error per body")
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{args.out}/scatter_field_err_vs_step.png", dpi=110)

    R = load(args.res, "D")
    base = load(args.res, "A").get("baseline")
    if base:
        R["baseline(A-trained)"] = base
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for name, r in R.items():
        if "sweep" not in r:
            continue
        ks = sorted(r["sweep"], key=float)
        x = [float(k) for k in ks]
        axs[0].plot(x, [r["sweep"][k]["err"][2] for k in ks], "o-", label=name)
        axs[1].plot(x, [r["sweep"][k]["sep_rel_err"][2] for k in ks], "o-", label=name)
        axs[2].plot(x, [r["sweep"][k]["energy_drift_rel"] for k in ks], "o-", label=name)
    for ax, t in zip(axs, ("err@20 (position, sim units)", "|sep error| / d at step 100", "|dE/E| at step 100")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("initial separation d / h_leaf (h_leaf = 0.5, eps = 0.5)")
        ax.set_title(t)
        ax.axvline(1.0, color="gray", lw=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}/scatter_field_separation_sweep.png", dpi=110)

    R = load(args.res, "C")
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    names = [n for n in R if "probe" in R[n]]
    for ax, name in zip(axs.ravel(), names):
        pr = R[name]["probe"]
        for i, d in enumerate(pr["dists"]):
            tr = np.array(pr["delete"][i])
            ax.plot(tr / pr["a_truth_before"][i], label=f"d={d:g}")
        ax.axvline(pr["warm"] - 0.5, color="k", ls="--")
        ax.set_title(f"{name}  lag {pr['lag_steps']}", fontsize=8)
        ax.set_ylabel("inward accel / truth pre-deletion")
        ax.set_xlabel("step (star deleted at dashed line; truth drops to 0 immediately)")
    axs.ravel()[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{args.out}/scatter_field_delete_star.png", dpi=110)

    R = load(args.res, "D")
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, name in zip(axs, ("ss_rec", "ms_rec", "ml_rec", "ml_noquad")):
        if name not in R or "curl" not in R[name]:
            continue
        a = np.array(R[name]["curl"]["a_map"])
        n = a.shape[0]
        xs = np.linspace(-8, 8, n)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        ax.quiver(gx, gy, a[..., 0], a[..., 1], np.hypot(a[..., 0], a[..., 1]), scale=None)
        ax.set_title(f"{name} force on test mass near static star; curl/total {R[name]['curl']['curl_over_total']:.2f}", fontsize=8)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{args.out}/scatter_field_force_field.png", dpi=110)

    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for ax, name in zip(axs.ravel(), ("baseline", "ss_rec", "ms_rec", "ms_rec_g128", "ml_rec", "ml_norec", "ml_noquad", "ms_rec_nomomfix")):
        p = f"{args.res}/A_{name}_traj.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        for s, c in zip(range(3), ("C0", "C1", "C2")):
            ax.plot(z["truth"][s, :, 0, 0], z["truth"][s, :, 0, 1], c + "-", lw=1)
            ax.plot(z["model"][s, :, 0, 0], z["model"][s, :, 0, 1], c + "--", lw=1)
        ax.set_title(f"A {name}: truth solid, model dashed (body 0, 100 steps)", fontsize=8)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{args.out}/scatter_field_2body_trajectories.png", dpi=110)


if __name__ == "__main__":
    main()
