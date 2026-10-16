"""Frame grid + mp4 of held-out solar rollouts: rows = truth and each arm, columns = ticks. Rows are labelled with arm names only.
Usage: python scripts/render_fewshot_solar.py --npz results/fewshot_adapter/solar_held_rollouts.npz --scene 0 --out videos/fewshot_solar_s0"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TICKS = (0, 50, 100, 200, 300, 500, 700, 1000)
COL = {0: ("gold", 40), 1: ("tab:blue", 14), 2: ("tab:red", 8), 3: ("0.55", 2)}


def draw(ax, P, kind, lim):
    for k, (c, s) in COL.items():
        sel = kind == k
        ax.scatter(P[sel, 0], P[sel, 1], c=c, s=s, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lim", type=float, default=30.0)
    args = ap.parse_args()
    d = np.load(args.npz)
    kind = d["kind"]
    rows = {"truth": d["truth"][args.scene].astype(np.float32)}
    for k in d.files:
        if k.endswith("_P"):
            rows[k[:-2]] = d[k][args.scene].astype(np.float32)
    T = min(v.shape[0] for v in rows.values()) - 1
    ticks = [t for t in TICKS if t <= T]
    fig, ax = plt.subplots(len(rows), len(ticks), figsize=(2.1 * len(ticks), 2.2 * len(rows)))
    for i, (name, P) in enumerate(rows.items()):
        for j, t in enumerate(ticks):
            draw(ax[i, j], P[t], kind, args.lim)
            if i == 0:
                ax[i, j].set_title(f"tick {t}", fontsize=8)
            if j == 0:
                ax[i, j].set_ylabel(name, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=110)
    plt.close()
    import subprocess

    fig, ax = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 3))
    frames = range(0, T + 1, 5)
    for n, t in enumerate(frames):
        for a, (name, P) in zip(ax, rows.items()):
            a.clear()
            draw(a, P[t], kind, args.lim)
            a.set_title(f"{name} t={t}", fontsize=8)
        fig.savefig(f"{args.out}_{n:04d}.png", dpi=80)
    plt.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", f"{args.out}_%04d.png", "-pix_fmt", "yuv420p", args.out + ".mp4"], check=True)
    subprocess.run(f"rm -f {args.out}_[0-9][0-9][0-9][0-9].png", shell=True)


if __name__ == "__main__":
    main()
