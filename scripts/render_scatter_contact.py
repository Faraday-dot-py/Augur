"""Diagnostic grid / side-by-side video (truth | model) for scripts/scatter_contact_scenes.py outputs. Local, streams frames to ffmpeg.

  python scripts/render_scatter_contact.py --family disc --tag B [--video] [--scene 0]
Reads results/contact/<family>_<tag>_traj.npz; writes videos/contact_<family>_<tag>_grid.png (and .mp4).
"""
import argparse
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

R = "results/contact"


def style(ax, lim, prims):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for pr in prims:
        k = pr["kind"]
        if k == "hbox":
            ax.add_patch(Rectangle(pr["lo"], pr["hi"][0] - pr["lo"][0], pr["hi"][1] - pr["lo"][1], fill=False, ec="k", lw=1.5))
        elif k == "hdisc":
            ax.add_patch(Circle(pr["c"], pr["r"], fill=False, ec="k", lw=1.5))
        elif k == "circle":
            ax.add_patch(Circle(pr["c"], pr["r"], fc="0.6", ec="k", lw=1))
        else:
            ax.add_patch(Polygon(np.array(pr["pts"]), fc="0.6", ec="k", lw=1))


def draw(ax, P, rad, n, color, filled):
    for p, r in zip(P[:n], rad[:n]):
        ax.add_patch(Circle(p, r, fc=color if filled else "none", ec=color, alpha=0.5 if filled else 1.0, lw=1))


def grid(truth, model, mask, rad, prims, ticks, lim, out, title, scenes=3):
    S = min(scenes, truth.shape[0])
    fig, axes = plt.subplots(S, len(ticks) + 1, figsize=(3 * (len(ticks) + 1), 3 * S))
    axes = np.atleast_2d(axes)
    for i in range(S):
        n = int(mask[i].sum())
        for j, t in enumerate(ticks):
            ax = axes[i, j]
            style(ax, lim, prims[i])
            draw(ax, truth[i, t], rad[i], n, "tab:blue", True)
            draw(ax, model[i, t], rad[i], n, "tab:red", False)
            ax.set_title(f"scene {i} tick {t} (blue=truth, red=model)", fontsize=7)
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


def video(truth, model, n, rad, prims, lim, out, fps=15):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), dpi=80)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{int(fig.get_figwidth() * fig.dpi)}x{int(fig.get_figheight() * fig.dpi)}",
           "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", out]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(truth.shape[0]):
        for ax, P, nm in ((axes[0], truth[t], "truth"), (axes[1], model[t], "model")):
            ax.clear()
            style(ax, lim, prims)
            draw(ax, P, rad, n, "tab:blue" if nm == "truth" else "tab:red", True)
            ax.set_title(f"{nm}  tick {t}", fontsize=9)
        fig.canvas.draw()
        pipe.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--scene", type=int, default=0)
    a = ap.parse_args()
    z = np.load(f"{R}/{a.family}_{a.tag}_traj.npz", allow_pickle=True)
    prims = json.loads(str(z["prims"]))
    base = f"videos/contact_{a.family}_{a.tag}"
    lim = 11
    grid(z["truth"], z["model"], z["mask"], z["rad"], prims, (0, 10, 30, 60, 100), lim, base + "_grid.png", f"{a.family} {a.tag}: truth vs model (gray/black = static obstacle)")
    if a.video:
        i = a.scene
        video(z["truth"][i], z["model"][i], int(z["mask"][i].sum()), z["rad"][i], prims[i], lim, base + ".mp4")


if __name__ == "__main__":
    main()
