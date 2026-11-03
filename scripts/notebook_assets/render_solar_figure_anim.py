import io
import os
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

INK, BRICK, PAPER = "#1f1b17", "#a43a2b", "#eadfc4"
OUT = "docs/design/assets/solar_figure_anim.mp4"
STILL = "docs/design/assets/solar_figure_still.png"
W, H, SS, LIM, PW, T1, TRAIL, FPS, SUB = 1280, 900, 2, 32.0, 0.6, 1000, 120, 30, 4
LOOP, XF = 45 * FPS, FPS
NF = LOOP + XF
PANEL, GH = 640, 260
RMAX = 32.0
CRF = int(os.environ.get("CRF", 24))


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


def squash(P):
    r = np.linalg.norm(P, axis=-1, keepdims=True)
    return P * (LIM * (np.minimum(r, 1e3) / LIM) ** PW / np.maximum(r, 1e-9)) * (r <= 1e3) + P * (r > 1e3)


def frames():
    d = np.load("results/solar/showcase_solar_orbit_ft2_kpadext_traj.npz")
    truth, model, kind, mass = d["truth"].astype(np.float64), d["model_orbit_ft2"].astype(np.float64), d["kind"], d["mass"]
    pl, moon, belt = np.where(kind == 1)[0], np.where(kind == 2)[0], np.where(kind == 3)[0]
    ts = np.arange(T1 + 1, dtype=float)
    rad = {n: np.linalg.norm(P[:T1 + 1][:, pl] - P[:T1 + 1][:, :1], axis=-1) for n, P in (("t", truth), ("m", model))}
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100 * SS, facecolor=PAPER)
    ax = [fig.add_axes([x / W, (H - PANEL) / H, PANEL / W, PANEL / H]) for x in (0, PANEL)]
    gx = [fig.add_axes([(x + 78) / W, 52 / H, (PANEL - 98) / W, (GH - 88) / H]) for x in (0, PANEL)]

    def trail(a, P, i, tt, color, lw, amax):
        xy = P[:, i]
        seg = np.stack([xy[:-1], xy[1:]], 1)
        al = np.linspace(0, amax, len(seg)) ** 1.5
        rgba = np.tile(matplotlib.colors.to_rgba(color), (len(seg), 1))
        rgba[:, 3] = al
        a.add_collection(LineCollection(seg, colors=rgba, linewidths=lw, capstyle="butt"))

    for f in range(NF):
        t = f * T1 / (NF - 1)
        tt = np.linspace(max(0.0, t - TRAIL), t, max(2, int((t - max(0.0, t - TRAIL)) * SUB) + 1))
        M, Q = squash(interp(model, tt)), squash(interp(truth, tt))
        for a in ax:
            a.cla(); a.set_xlim(-LIM, LIM); a.set_ylim(-LIM, LIM); a.set_aspect("equal"); a.axis("off"); a.set_facecolor(PAPER)
        for i in pl:
            trail(ax[0], Q, i, tt, INK, 2.2, 0.9)
            ax[0].plot(Q[-1, i, 0], Q[-1, i, 1], "o", color=INK, ms=7 + 70 * mass[i], mew=0)
        for i in moon:
            trail(ax[0], Q, i, tt, INK, 1.6, 0.9)
            ax[0].plot(Q[-1, i, 0], Q[-1, i, 1], "o", color=INK, mfc=PAPER, ms=7, mew=1.8)
        b = Q[-1][belt]
        ib = np.abs(b).max(-1) < 1e3
        ax[0].plot(b[ib, 0], b[ib, 1], ".", color=INK, ms=4.5, alpha=0.7, mew=0)
        ax[0].plot(Q[-1, 0, 0], Q[-1, 0, 1], "o", color=INK, mfc="none", ms=26, mew=2.4)
        ax[0].plot(Q[-1, 0, 0], Q[-1, 0, 1], "o", color=INK, ms=13, mew=0)
        P = M[-1]
        b = P[belt]
        ib = np.abs(b).max(-1) < 1e3
        ax[1].plot(b[ib, 0], b[ib, 1], ".", color=BRICK, ms=4.5, alpha=0.75, mew=0)
        for i in pl:
            trail(ax[1], M, i, tt, BRICK, 2.6, 0.85)
            ax[1].plot(P[i, 0], P[i, 1], "o", color=BRICK, ms=7 + 70 * mass[i], mew=0)
        for i in moon:
            trail(ax[1], M, i, tt, BRICK, 1.6, 0.9)
            ax[1].plot(P[i, 0], P[i, 1], "o", color=BRICK, mfc=PAPER, ms=7, mew=1.8)
        ax[1].plot(P[0, 0], P[0, 1], "o", color=INK, mfc="none", ms=26, mew=2.4)
        ax[1].plot(P[0, 0], P[0, 1], "o", color=INK, ms=13, mew=0)
        n = int(np.floor(t)) + 1
        for g, key, col in ((gx[0], "t", INK), (gx[1], "m", BRICK)):
            g.cla(); g.set_facecolor(PAPER)
            g.set_xlim(0, T1); g.set_ylim(0, RMAX)
            for s in g.spines.values():
                s.set_visible(False)
            g.spines["left"].set_visible(True); g.spines["bottom"].set_visible(True)
            g.spines["left"].set_color(INK); g.spines["bottom"].set_color(INK)
            g.set_xticks([0, 500, 1000]); g.set_yticks([0, 15, 30])
            g.tick_params(colors=INK, labelsize=17, length=4, width=1.4)
            for j in range(rad[key].shape[1]):
                g.plot(ts[:n], rad[key][:n, j], color=col, lw=1.8, alpha=0.85)
            g.set_xlabel('time (ticks)', color=INK, fontsize=17, labelpad=2)
            g.set_ylabel('distance', color=INK, fontsize=17, labelpad=2)
            g.axvline(t, color=INK, lw=1.2, ls=(0, (4, 3)))
            g.text(min(t + 12, T1 - 150), RMAX * 0.93, f"tick {int(round(t))}", color=INK, fontsize=18, va="top")
        buf = io.BytesIO()
        fig.savefig(buf, format="rgba", dpi=100 * SS, facecolor=PAPER)
        a = np.frombuffer(buf.getvalue(), np.uint8).reshape(H * SS, W * SS, 4)[..., :3]
        yield np.asarray(Image.fromarray(a).resize((W, H), Image.LANCZOS))
    plt.close(fig)


def main():
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", "-an", OUT]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    head, n = [], 0
    for f, fr in enumerate(frames()):
        if f == 0:
            Image.fromarray(fr).save(STILL)
        if f < XF:
            head.append(fr)
        if XF <= f < NF - XF:
            enc.stdin.write(fr.tobytes()); n += 1
        elif f >= NF - XF:
            a = (f - (NF - XF) + 1) / (XF + 1)
            b = ((1 - a) * fr.astype(np.float32) + a * head[f - (NF - XF)]).round().astype(np.uint8)
            enc.stdin.write(b.tobytes()); n += 1
    enc.stdin.close(); enc.wait()
    print("frames", n, "mp4", os.path.getsize(OUT))


if __name__ == "__main__":
    main()
