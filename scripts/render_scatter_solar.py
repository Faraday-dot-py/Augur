"""Side-by-side video (exact sim | scatter-field model) of the solar-system scene with trails, plus planet-radius-vs-time strips. Local, streams frames to ffmpeg.

  python scripts/render_scatter_solar.py --traj results/solar/solar_traj.npz --model orbit_ft2 --out videos/scatter_solar_orbit_ft2.mp4 [--every 4] [--trail 150] [--grid out.png]
"""
import argparse
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--trail", type=int, default=150)
ap.add_argument("--lim", type=float, default=26.0)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--grid", default=None)
args = ap.parse_args()

d = np.load(args.traj)
truth, model, kind, mass = d["truth"], d["model_" + args.model], d["kind"], d["mass"]
T = truth.shape[0] - 1
pl = np.where(kind == 1)[0]
moon = np.where(kind == 2)[0]
belt = np.where(kind == 3)[0]
col = plt.get_cmap("tab10")(np.arange(len(pl)) % 10)
R = lambda P: np.linalg.norm(P[:, pl] - P[:, :1], axis=-1)
rt, rm = R(truth), R(model)


def panel(ax, P, t, name):
    ax.clear()
    ax.set_facecolor("black")
    ax.set_xlim(-args.lim, args.lim)
    ax.set_ylim(-args.lim, args.lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    inb = np.abs(P[t, belt]).max(-1) < 1e3
    ax.plot(P[t, belt][inb, 0], P[t, belt][inb, 1], ".", color="0.55", ms=1.5)
    lo = max(0, t - args.trail)
    for j, i in enumerate(pl):
        ax.plot(P[lo:t + 1, i, 0], P[lo:t + 1, i, 1], color=col[j], lw=1, alpha=0.7)
        ax.plot(P[t, i, 0], P[t, i, 1], "o", color=col[j], ms=3 + 40 * mass[i])
    for i in moon:
        ax.plot(P[lo:t + 1, i, 0], P[lo:t + 1, i, 1], color="w", lw=0.6, alpha=0.8)
        ax.plot(P[t, i, 0], P[t, i, 1], "o", color="w", ms=2)
    ax.plot(P[lo:t + 1, 0, 0], P[lo:t + 1, 0, 1], color="gold", lw=0.8)
    ax.plot(P[t, 0, 0], P[t, 0, 1], "o", color="gold", ms=9)
    ax.set_title(f"{name}  tick {t}  (t = {t * 0.1:.0f})", color="w", fontsize=11)


fig = plt.figure(figsize=(12, 8), dpi=90, facecolor="black")
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.12, wspace=0.04, left=0.03, right=0.97, top=0.95, bottom=0.07)
axt, axm = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axr = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
w, h = int(fig.get_figwidth() * fig.dpi), int(fig.get_figheight() * fig.dpi)


def strips(t):
    for ax, r, name in ((axr[0], rt, "exact"), (axr[1], rm, "model")):
        ax.clear()
        ax.set_facecolor("black")
        for j in range(len(pl)):
            ax.plot(r[:, j], color=col[j], lw=0.8)
        ax.axvline(t, color="w", lw=0.8)
        ax.set_ylim(0, 40)
        ax.set_xlim(0, T)
        ax.set_ylabel(f"{name} r / star", color="w", fontsize=8)
        ax.tick_params(colors="w", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("0.4")


def draw(t):
    panel(axt, truth, t, "exact sim")
    panel(axm, model, t, f"scatter field ({args.model})")
    strips(t)


if args.grid:
    ticks = [0, T // 8, T // 4, T // 2, T]
    f2, axes = plt.subplots(2, len(ticks), figsize=(3.2 * len(ticks), 6.6), facecolor="black")
    for j, t in enumerate(ticks):
        panel(axes[0, j], truth, t, "exact")
        panel(axes[1, j], model, t, "model")
    f2.savefig(args.grid, dpi=80, facecolor="black")
    print("wrote", args.grid)
else:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", args.out]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(0, T + 1, args.every):
        draw(t)
        fig.canvas.draw()
        pipe.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    pipe.stdin.close()
    assert pipe.wait() == 0
    print("wrote", args.out)
