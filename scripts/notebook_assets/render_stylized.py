import shutil
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, BRICK, CREAM = "#1f1b17", "#a43a2b", "#f3ead6"
OUT = "docs/design/assets/"
BH_SRC = "videos/showcase_black_hole_1e9_lensed_300ticks_kpadext_lattice_smoothed.mp4"


def solar(t=1000, lim=26.0, trail=150):
    d = np.load("results/solar/showcase_solar_orbit_ft2_kpadext_traj.npz")
    truth, model, kind, mass = d["truth"], d["model_orbit_ft2"], d["kind"], d["mass"]
    pl, moon, belt = np.where(kind == 1)[0], np.where(kind == 2)[0], np.where(kind == 3)[0]
    T = truth.shape[0] - 1
    R = lambda P: np.linalg.norm(P[:, pl] - P[:, :1], axis=-1)
    fig = plt.figure(figsize=(9, 6.4), dpi=200)
    fig.patch.set_alpha(0)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], hspace=0.08, wspace=0.1, left=0.02, right=0.98, top=0.99, bottom=0.07)
    for k, (P, c) in enumerate(((truth, INK), (model, BRICK))):
        ax = fig.add_subplot(gs[0, k])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0)
        inb = np.abs(P[t, belt]).max(-1) < 1e3
        ax.plot(P[t, belt][inb, 0], P[t, belt][inb, 1], ".", color=c, ms=1.6, alpha=0.6, mew=0)
        lo = max(0, t - trail)
        for i in pl:
            ax.plot(P[lo:t + 1, i, 0], P[lo:t + 1, i, 1], color=c, lw=0.7, alpha=0.5)
            ax.plot(P[t, i, 0], P[t, i, 1], "o", color=c, ms=3 + 40 * mass[i], mew=0)
        for i in moon:
            ax.plot(P[lo:t + 1, i, 0], P[lo:t + 1, i, 1], color=c, lw=0.5, alpha=0.7)
            ax.plot(P[t, i, 0], P[t, i, 1], "o", color=c, mfc="none", ms=3.5, mew=0.8)
        ax.plot(P[t, 0, 0], P[t, 0, 1], "o", color=c, mfc="none", ms=11, mew=1.4)
        ax.plot(P[t, 0, 0], P[t, 0, 1], "o", color=c, ms=5, mew=0)
    for k, (P, c) in enumerate(((truth, INK), (model, BRICK))):
        ax = fig.add_subplot(gs[1, k])
        ax.patch.set_alpha(0)
        r = R(P)
        for j in range(len(pl)):
            ax.plot(r[:, j], color=c, lw=0.6, alpha=0.8)
        ax.axvline(t, color=c, lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(0, 40); ax.set_xlim(0, T)
        ax.set_yticks([0, 20, 40]); ax.set_xticks([0, 1000, 2000])
        ax.tick_params(colors=INK, labelsize=7, length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.savefig(OUT + "solar_stylized.png", dpi=200, transparent=True)
    plt.close(fig)


def bh():
    lut = np.zeros((256, 3), np.float32)
    stops = [(0.0, "#1f1b17"), (0.5, "#a43a2b"), (1.0, "#f3ead6")]
    cols = [np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) for _, h in stops]
    xs = [s for s, _ in stops]
    for i in range(256):
        x = i / 255
        for c in range(3):
            lut[i, c] = np.interp(x, xs, [col[c] for col in cols])
    lut = lut.astype(np.uint8)
    S, N = 1080, 540
    rd = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", BH_SRC, "-vf", f"crop=1008:1008:36:72,scale={N}:{N}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    fr = []
    first = rd.stdout.read(N * N)
    g = np.frombuffer(first, np.uint8)
    lo, hi = np.percentile(g, 0.5), 255.0
    wr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{N}x{N}", "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT + "bh_1e9_stylized.mp4"], stdin=subprocess.PIPE)
    n = 0
    buf = first
    while len(buf) == N * N:
        g = np.frombuffer(buf, np.uint8).astype(np.float32)
        g = np.clip((g - lo) / (hi - lo), 0, 1)
        wr.stdin.write(lut[(g * 255).astype(np.uint8)].tobytes())
        if n == 150:
            from PIL import Image
            Image.fromarray(lut[(g * 255).astype(np.uint8)].reshape(N, N, 3)).save(OUT + "bh_1e9_stylized_poster.jpg", quality=90)
        n += 1
        buf = rd.stdout.read(N * N)
    wr.stdin.close(); wr.wait(); rd.wait()
    print("frames", n)


if __name__ == "__main__":
    what = sys.argv[1:] or ["solar", "bh", "tiling"]
    if "solar" in what: solar()
    if "bh" in what: bh()
    if "tiling" in what: shutil.copy(OUT + "zfold_tiles.png", OUT + "tiling_stylized.png")
