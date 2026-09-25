"""Plots for the interpretability test from results/interp_*.json / interp_maps_maps.npz -> docs/debugging/figures/interp_*.png (local, tiny)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("docs/debugging/figures", exist_ok=True)


def rho_fig():
    r = json.load(open("results/interp_stage1.json"))["rho_sweep"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for a, (k, v) in zip(ax.flat, r.items()):
        rh = np.array(v["rhos"])
        i = 1
        a.plot(rh, v["truth"][i], "ko-", label="exact label")
        a.plot(rh, v["taylor"][i], "b--", label="2nd-order Taylor")
        a.plot(rh, v["est"][i], "r^-", label="estimator")
        a.set_xscale("log")
        a.set_title(f"{k}  slope truth/est {v['truth_slope'][i]:.2f}/{v['est_slope'][i]:.2f}")
        a.set_xlabel("rho = size/dist")
        a.set_ylabel("mean log label")
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig("docs/debugging/figures/interp_rho.png", dpi=110)


def maps_fig():
    m = np.load("results/interp_maps_maps.npz")
    names = ["analytic", "yukawa30", "inv_distance"]
    rows = [("density", None), ("est_msize", "log2 mean accepted size (est)"), ("geoM_msize", "log2 mean accepted size (geo, matched cost)"),
            ("est_nacc", "accepted pairs / particle (est)"), ("est_prederr", "predicted rel. error (est)")]
    fig, ax = plt.subplots(len(rows), 3, figsize=(12, 3.6 * len(rows)))
    for j, n in enumerate(names):
        for i, (k, t) in enumerate(rows):
            arr = m["density"] if k == "density" else m[f"{n}_{k}"]
            if k == "density":
                arr = np.log10(np.maximum(arr, 1))
            elif k == "est_prederr":
                arr = np.log10(np.maximum(arr, 1e-12))
            im = ax[i, j].imshow(arr.T, origin="lower", cmap="viridis")
            ax[i, j].set_title(f"{n}: {'log10 density' if k == 'density' else t}", fontsize=8)
            ax[i, j].axis("off")
            fig.colorbar(im, ax=ax[i, j], fraction=0.046)
    fig.tight_layout()
    fig.savefig("docs/debugging/figures/interp_maps.png", dpi=90)


if __name__ == "__main__":
    for f in (rho_fig, maps_fig):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
