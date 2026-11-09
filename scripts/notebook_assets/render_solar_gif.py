import io
import os
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

INK, BRICK, PAPER = "#1f1b17", "#a43a2b", "#eadfc4"
OUT = "docs/design/assets/solar_bg.gif"
PX, SS, LIM, NF, XF, T1, TRAIL, FPS, SUB, NCOL, OP = 720, 2, 15.0, 1155, 25, 1000, 120, 25, 4, 24, 512


def interp(P, t):
    t = np.clip(np.asarray(t, float), 0, len(P) - 1.0001)
    k = np.floor(t).astype(int)
    u = (t - k)[:, None, None]
    n = len(P)
    p0, p1 = P[k], P[k + 1]
    m0 = (P[k + 1] - P[np.maximum(k - 1, 0)]) / (1 + (k > 0))[:, None, None]
    m1 = (P[np.minimum(k + 2, n - 1)] - P[k]) / (1 + (k + 1 < n - 1))[:, None, None]
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


def frames():
    d = np.load("results/solar/showcase_solar_orbit_ft2_kpadext_traj.npz")
    truth, model, kind, mass = d["truth"].astype(np.float64), d["model_orbit_ft2"].astype(np.float64), d["kind"], d["mass"]
    pl, moon, belt = np.where(kind == 1)[0], np.where(kind == 2)[0], np.where(kind == 3)[0]
    S = PX * SS
    fig = plt.figure(figsize=(PX / 100, PX / 100), dpi=100 * SS, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])

    def trail(P, i, tt, color, lw, amax):
        xy = P[:, i]
        seg = np.stack([xy[:-1], xy[1:]], 1)
        a = np.linspace(0, amax, len(seg)) ** 1.5
        rgba = np.tile(matplotlib.colors.to_rgba(color), (len(seg), 1))
        rgba[:, 3] = a
        ax.add_collection(LineCollection(seg, colors=rgba, linewidths=lw, capstyle="butt"))

    for f in range(NF):
        t = f * T1 / (NF - 1)
        tt = np.linspace(max(0.0, t - TRAIL), t, max(2, int((t - max(0.0, t - TRAIL)) * SUB) + 1))
        M, Q = interp(model, tt), interp(truth, tt)
        P = M[-1]
        ax.cla()
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal"); ax.axis("off")
        ax.set_facecolor(PAPER)
        for i in pl:
            trail(Q, i, tt, INK, 0.8, 0.18)
        inb = np.abs(P[belt]).max(-1) < 1e3
        ax.plot(P[belt][inb, 0], P[belt][inb, 1], ".", color=BRICK, ms=2.4, alpha=0.75, mew=0)
        for i in pl:
            trail(M, i, tt, BRICK, 1.6, 0.85)
            ax.plot(P[i, 0], P[i, 1], "o", color=BRICK, ms=4 + 45 * mass[i], mew=0)
        for i in moon:
            trail(M, i, tt, BRICK, 1.0, 0.9)
            ax.plot(P[i, 0], P[i, 1], "o", color=BRICK, mfc=PAPER, ms=4, mew=1.2)
        ax.plot(P[0, 0], P[0, 1], "o", color=INK, mfc="none", ms=17, mew=1.8)
        ax.plot(P[0, 0], P[0, 1], "o", color=INK, ms=8, mew=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="rgba", dpi=100 * SS, facecolor=PAPER)
        a = np.frombuffer(buf.getvalue(), np.uint8).reshape(S, S, 4)[..., :3]
        yield np.asarray(Image.fromarray(a).resize((OP, OP), Image.LANCZOS))
    plt.close(fig)


def main():
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
    mkv, pal = os.path.join(tmp, "raw.mkv"), os.path.join(tmp, "pal.png")
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OP}x{OP}",
                            "-r", str(FPS), "-i", "-", "-c:v", "ffv1", mkv], stdin=subprocess.PIPE)
    head, n = [], 0
    for f, fr in enumerate(frames()):
        if f < XF:
            head.append(fr)
        if XF <= f < NF - XF:
            enc.stdin.write(fr.tobytes()); n += 1
        elif f >= NF - XF:
            a = (f - (NF - XF) + 1) / (XF + 1)
            b = ((1 - a) * fr.astype(np.float32) + a * head[f - (NF - XF)]).round().astype(np.uint8)
            enc.stdin.write(b.tobytes()); n += 1
    enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mkv, "-vf", f"palettegen=max_colors={NCOL}:stats_mode=full:reserve_transparent=0", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mkv, "-i", pal, "-lavfi", "paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", OUT], check=True)
    print("frames", n, "gif", os.path.getsize(OUT))


if __name__ == "__main__":
    main()
