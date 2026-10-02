"""Diagnostic grid for scatter-field 2-body rollouts. Usage:
python scripts/render_scatter_rollout_grid.py --npz <A_x_traj.npz> --out videos/x.png"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(args.npz)
    M, T = d["model"], d["truth"]
    S = M.shape[0]
    fig, ax = plt.subplots(S, 3, figsize=(13, 3.4 * S))
    for s in range(S):
        a = ax[s, 0]
        for b, c in ((0, "C0"), (1, "C1")):
            a.plot(T[s, :, b, 0], T[s, :, b, 1], c, lw=2, alpha=0.5, label=f"truth body {b}")
            a.plot(M[s, :, b, 0], M[s, :, b, 1], c + "--", lw=1, label=f"model body {b}")
            a.plot(*T[s, 0, b], "ko", ms=4)
        a.set_aspect("equal")
        a.set_title(f"scene {s} paths (dot = step 0)")
        if s == 0:
            a.legend(fontsize=6)
        st = np.linalg.norm(T[s, :, 0] - T[s, :, 1], axis=-1)
        sm = np.linalg.norm(M[s, :, 0] - M[s, :, 1], axis=-1)
        ax[s, 1].plot(st, label="truth")
        ax[s, 1].plot(sm, "--", label="model")
        ax[s, 1].set_title("separation vs step")
        ax[s, 1].legend(fontsize=6)
        e = np.linalg.norm(M[s] - T[s], axis=-1).mean(1)
        ax[s, 2].semilogy(e + 1e-9)
        ax[s, 2].set_title("mean position error vs step")
    plt.tight_layout()
    plt.savefig(args.out, dpi=90)


if __name__ == "__main__":
    main()
