"""Diagnostic grid for large-N scatter-field rollouts (reference vs model density at fixed ticks + error curve).
Usage: python scripts/render_scaling_grid.py --npz results/scatter_scaling_dump_100000.npz --out videos/scatter_scaling_1e5_grid.png"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=128)
    args = ap.parse_args()
    d = np.load(args.npz)
    M, T, ticks, err, L = d["model"], d["truth"], d["ticks"], d["err"], float(d["extent"])
    n = M.shape[1]
    nt = len(ticks)
    fig, ax = plt.subplots(3, nt, figsize=(2.6 * nt, 8))
    rng = [[-L / 2, L / 2], [-L / 2, L / 2]]
    for j, t in enumerate(ticks):
        for i, (name, X) in enumerate((("reference", T), ("model", M))):
            h, _, _ = np.histogram2d(X[j, :, 0], X[j, :, 1], bins=args.bins, range=rng)
            ax[i, j].imshow(np.log10(h.T + 1), origin="lower", extent=[-L / 2, L / 2, -L / 2, L / 2], cmap="magma", vmin=0, vmax=np.log10(max(2, T[0].shape[0] / 30)))
            ax[i, j].set_title(f"{name} tick {t}", fontsize=8)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        e = np.linalg.norm(M[j] - T[j], axis=-1)
        ax[2, j].hist(np.log10(e + 1e-6), bins=40, color="C0")
        ax[2, j].set_title(f"log10 per-body err, tick {t}", fontsize=8)
    fig.suptitle(f"N = {n}: log10 density (grid {args.bins}^2, box {L:g}); bottom row = per-body position error histogram", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    fig2, a2 = plt.subplots(figsize=(5, 3.5))
    a2.semilogy(err[1:] + 1e-9)
    a2.set_xlabel("tick")
    a2.set_ylabel("mean position error")
    a2.set_title(f"N={n} mean err vs tick")
    plt.tight_layout()
    plt.savefig(args.out.replace(".png", "_err.png"), dpi=110)


if __name__ == "__main__":
    main()
