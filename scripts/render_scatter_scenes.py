"""Diagnostic grids and side-by-side videos (truth | model) for scripts/scatter_scenes.py outputs. Local, streams frames to ffmpeg.

  python scripts/render_scatter_scenes.py --scene bounce --tag zeroshot [--video]
Reads results/scenes/<scene>_<tag>{.json,_traj.npz}; writes videos/scenes_<scene>_<tag>_grid.png (and .mp4).
"""
import argparse
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

R = "results/scenes"


def style(ax, lim, box=False):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if box:
        ax.add_patch(Rectangle((-7, -7), 14, 14, fill=False, ec="k", lw=1))


def draw(ax, P, mask, radius, color, filled):
    n = int(mask.sum()) if mask is not None else P.shape[0]
    if radius:
        for p in P[:n]:
            ax.add_patch(Circle(p, radius, fc=color if filled else "none", ec=color, alpha=0.5 if filled else 1.0, lw=1))
    else:
        ax.plot(P[:n, 0], P[:n, 1], ".", color=color, ms=2, alpha=0.7)


def snap_grid(truth, model, mask, ticks, lim, out, title, radius=None, box=False, scenes=3):
    S = min(scenes, truth.shape[0])
    fig, axes = plt.subplots(S, len(ticks) + 1, figsize=(3 * (len(ticks) + 1), 3 * S))
    axes = np.atleast_2d(axes)
    for i in range(S):
        mk = mask[i] if mask is not None else None
        for j, t in enumerate(ticks):
            ax = axes[i, j]
            style(ax, lim, box)
            draw(ax, truth[i, t], mk, radius, "tab:blue", True)
            draw(ax, model[i, t], mk, radius, "tab:red", False)
            ax.set_title(f"scene {i} tick {t} (blue=truth, red=model)", fontsize=7)
        n = int(mk.sum()) if mk is not None else truth.shape[2]
        e = np.linalg.norm(model[i, :, :n] - truth[i, :, :n], axis=-1).mean(1)
        ax = axes[i, -1]
        ax.semilogy(np.maximum(e, 1e-5))
        ax.set_title("mean position error vs tick", fontsize=7)
        ax.set_xlabel("tick", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print("wrote", out)


def video(truth, model, mask, lim, out, radius=None, box=False, fps=20, every=1):
    T = truth.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), dpi=80)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{int(fig.get_figwidth() * fig.dpi)}x{int(fig.get_figheight() * fig.dpi)}",
           "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", out]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(0, T, every):
        for ax, P, nm in ((axes[0], truth[t], "truth"), (axes[1], model[t], "model")):
            ax.clear()
            style(ax, lim, box)
            draw(ax, P, mask, radius, "tab:blue" if nm == "truth" else "tab:red", True)
            ax.set_title(f"{nm}  tick {t * every}", fontsize=9)
        fig.canvas.draw()
        pipe.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    npz = np.load(f"{R}/{a.scene}_{a.tag}_traj.npz", allow_pickle=True)
    js = json.load(open(f"{R}/{a.scene}_{a.tag}.json"))
    base = f"videos/scenes_{a.scene}_{a.tag}"
    if a.scene == "bounce":
        t, m, mk = npz["truth"], npz["model"], npz["mask"]
        snap_grid(t, m, mk, (0, 10, 30, 60, 100), 9, base + "_grid.png", f"bounce {a.tag}: truth vs model, box [-7,7], r=.75", radius=0.75, box=True)
        if a.video:
            video(t[0], m[0], mk[0], 9, base + ".mp4", radius=0.75, box=True, fps=15)
    elif a.scene == "bh":
        t, m = npz["truth"][None], npz["model"][None]
        snap_grid(t, m, None, (0, 20, 40, 60, 100), 30, base + "_grid.png", f"black hole N=300 c=10 {a.tag}", scenes=1)
        if a.video:
            video(t[0], m[0], None, 30, base + ".mp4", fps=15)
    elif a.scene == "globular":
        t, m = npz["truth"][None], npz["model"][None]
        snap_grid(t, m, None, (0, 20, 50, 100), 20, base + "_grid.png", f"globular N=300 {a.tag} (every 10 ticks, index shown = tick/10)", scenes=1)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        run = js["runs"]["300"][0]
        for ax, k in zip(axes, ("r10", "r50", "r90", "e")):
            for nm, c in (("truth", "tab:blue"), ("twin", "tab:green"), ("model", "tab:red")):
                s = run["stats_" + nm]
                ax.plot(s["t"], s[k], color=c, label=nm)
            ax.set_title(k)
            ax.set_xlabel("tick")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(base + "_stats.png", dpi=90)
        if a.video:
            video(t[0], m[0], None, 20, base + ".mp4", fps=15)
    elif a.scene == "orbit":
        for fam in ("binary", "planetary"):
            t, m = npz[fam + "_truth"], npz[fam + "_model"]
            mk = npz[fam + "_mask"]
            S = 3
            fig, axes = plt.subplots(S, 2, figsize=(9, 4 * S))
            for i in range(S):
                n = int(mk[i].sum())
                ax = axes[i, 0]
                for b in range(n):
                    ax.plot(t[i, :, b, 0], t[i, :, b, 1], color=f"C{b}", lw=3, alpha=0.3)
                    ax.plot(m[i, :, b, 0], m[i, :, b, 1], color=f"C{b}", lw=1, ls="--")
                ax.set_aspect("equal")
                ax.set_title(f"{fam} scene {i}: thick = truth, dashed = model, 1000 ticks", fontsize=8)
                ax = axes[i, 1]
                for b in range(1, n):
                    rt = np.linalg.norm(t[i, :, b] - t[i, :, 0], axis=-1)
                    rm = np.linalg.norm(m[i, :, b] - m[i, :, 0], axis=-1)
                    ax.plot(rt, color=f"C{b}", lw=3, alpha=0.3)
                    ax.plot(rm, color=f"C{b}", lw=1, ls="--")
                ax.set_title("distance to body 0 vs tick", fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{base}_{fam}_grid.png", dpi=90)
            plt.close(fig)
            print("wrote", f"{base}_{fam}_grid.png")
            if a.video and fam == "planetary":
                video(t[0], m[0], mk[0], 22, base + "_planetary.mp4", fps=30, every=5)


if __name__ == "__main__":
    main()
