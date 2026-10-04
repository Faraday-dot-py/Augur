"""Diagnostic grid for scatter_ood trajectories. Usage:
python scripts/render_scatter_ood_grid.py --npz results/scatter_ood_traj.npz --keys axis/name,axis/name --out videos/x.png [--mp4 videos/x.mp4]
Per row (one scene): col 1 end-state positions truth (grey) vs model (red) at three times; col 2 paths of up to 12 bodies (thick faint = reference, dashed = model);
col 3 mean position error vs time (log) with the constant-velocity coasting error and a 1e-5 perturbation floor."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mp4", default="")
    args = ap.parse_args()
    d = np.load(args.npz)
    keys = args.keys.split(",")
    fig, ax = plt.subplots(len(keys), 3, figsize=(14, 3.6 * len(keys)), squeeze=False)
    for r, k in enumerate(keys):
        M, T = d[k + "/model"], d[k + "/truth"]
        dt = float(d[k + "/dt"])
        nt = M.shape[0]
        a = ax[r, 0]
        for frac, col in ((0.0, "k"), (0.5, "C0"), (1.0, "C3")):
            i = int(frac * (nt - 1))
            a.scatter(*T[i].T, s=6, c=col, alpha=0.35)
            if col != "k":
                a.scatter(*M[i].T, s=10, facecolors="none", edgecolors=col, linewidths=0.6)
        a.set_aspect("equal")
        a.set_title(f"{k}\npositions at t=0 (black), {0.5 * (nt - 1) * dt:.1f} (blue), {(nt - 1) * dt:.1f} (red); dots truth, rings model", fontsize=7)
        a = ax[r, 1]
        for b in range(min(12, M.shape[1])):
            a.plot(T[:, b, 0], T[:, b, 1], lw=2, alpha=0.3, color=f"C{b % 10}")
            a.plot(M[:, b, 0], M[:, b, 1], "--", lw=0.8, color=f"C{b % 10}")
        a.set_aspect("equal")
        a.set_title("paths of first bodies (faint = reference, dashed = model)", fontsize=7)
        a = ax[r, 2]
        t = np.arange(nt) * dt
        a.semilogy(t, d[k + "/err"] + 1e-9, label="model")
        a.semilogy(t, d[k + "/ballistic"] + 1e-9, label="coasting")
        a.semilogy(t, d[k + "/floor"] + 1e-9, label="1e-5 perturbed truth")
        a.set_title("mean position error vs time", fontsize=7)
        a.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(args.out, dpi=85)
    if args.mp4:
        from matplotlib.animation import FFMpegWriter

        k = keys[0]
        M, T = d[k + "/model"], d[k + "/truth"]
        lim = max(10.0, float(np.abs(T).max()) * 1.05)
        lim = min(lim, 60.0)
        f, axs = plt.subplots(1, 2, figsize=(9, 4.5))
        w = FFMpegWriter(fps=20)
        with w.saving(f, args.mp4, 90):
            for i in range(0, M.shape[0], max(1, M.shape[0] // 200)):
                for a, X, ttl in ((axs[0], T, "reference"), (axs[1], M, "model")):
                    a.clear()
                    a.scatter(*X[i].T, s=6)
                    a.set_xlim(-lim, lim)
                    a.set_ylim(-lim, lim)
                    a.set_aspect("equal")
                    a.set_title(f"{k} {ttl} step {i}", fontsize=8)
                w.grab_frame()


if __name__ == "__main__":
    main()
