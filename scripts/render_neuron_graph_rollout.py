import argparse
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

V_LO, V_HI = -85.0, -45.0
TOL = 2


def layout(exc):
    n = len(exc)
    pos = np.zeros((n, 2))
    for kind, radius in ((True, 1.0), (False, 0.55)):
        idx = np.flatnonzero(exc == kind)
        ang = 2 * np.pi * np.arange(len(idx)) / max(1, len(idx)) + (0.0 if kind else np.pi / len(idx))
        pos[idx] = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return pos


def running_fscore(true_spk, pred_spk, t):
    ts, ps = true_spk[: t + 1], pred_spk[: t + 1]

    def near(a):
        out = np.zeros_like(a)
        for d in range(-TOL, TOL + 1):
            out |= np.roll(a, d, axis=0)
        return out

    if ts.sum() == 0 and ps.sum() == 0:
        return 1.0
    prec = (ps & near(ts)).sum() / max(1, ps.sum())
    rec = (ts & near(ps)).sum() / max(1, ts.sum())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("out")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--title", default="")
    a = p.parse_args()
    d = np.load(a.npz)
    truth, pred, abl = d["truth"][a.window][: a.steps], d["pred"][a.window][: a.steps], d["ablation"][a.window][: a.steps]
    weights, exc = d["weights"], d["exc"]
    n = len(exc)
    pos = layout(exc)
    ts, ps = truth[..., 3] > 0.5, pred[..., 3] > 0.5
    rmse = np.sqrt(((np.clip(pred[..., 0], -1000, 1000) - truth[..., 0]) ** 2).mean(axis=1))
    rmse_abl = np.sqrt(((np.clip(abl[..., 0], -1000, 1000) - truth[..., 0]) ** 2).mean(axis=1))
    dst, src = np.nonzero(weights)
    segs = [[pos[s], pos[t]] for s, t in zip(src, dst)]
    ecol = np.where(exc[src], 0.0, 1.0)
    edge_rgba = np.where(ecol[:, None] == 0.0, np.array([[0.9, 0.55, 0.3, 0.25 if n <= 20 else 0.05]]), np.array([[0.35, 0.6, 0.95, 0.25 if n <= 20 else 0.05]]))
    vec = pos[dst] - pos[src]
    tip = pos[dst] - 0.12 * vec
    base_size = 160 if n <= 20 else 34

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#101418")
    axes, scatters = [], []
    for i, (name, data) in enumerate((("TEACHER (truth)", truth), ("LEARNED MODEL (free rollout)", pred))):
        ax = fig.add_axes([0.02 + 0.49 * i, 0.30, 0.47, 0.53])
        ax.set_facecolor("#101418")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_collection(LineCollection(segs, colors=edge_rgba, linewidths=0.5, zorder=1))
        if n <= 20:
            ax.quiver(tip[:, 0], tip[:, 1], 0.1 * vec[:, 0], 0.1 * vec[:, 1], angles="xy", scale_units="xy", scale=1,
                      width=0.0012, headwidth=6, headlength=7, color=edge_rgba, zorder=2)
        for kind, marker in ((True, "o"), (False, "s")):
            m = exc == kind
            sc = ax.scatter(pos[m, 0], pos[m, 1], s=base_size, c=np.zeros(m.sum()), cmap="viridis", vmin=V_LO, vmax=V_HI,
                            marker=marker, edgecolors="#2a3138" if kind else "#d7dce2", linewidths=0.8 if kind else 1.4, zorder=3)
            scatters.append((i, m, sc))
        ax.set_title(name, color="#e8edf2", fontsize=14, pad=4)
        axes.append(ax)
    hdr = a.title or ""
    fig.text(0.5, 0.985, hdr, color="#e8edf2", ha="center", va="top", fontsize=13)
    fig.text(0.5, 0.935, "colour = membrane voltage (dark -85 mV to yellow -45 mV); white outline and larger node = spike this tick\n"
             "circle = excitatory (outer ring), square = inhibitory (inner ring); faint arrows = directed synapses (orange from E, blue from I)",
             color="#9aa5b1", ha="center", va="top", fontsize=9)
    readout = fig.text(0.5, 0.285, "", color="#e8edf2", ha="center", va="top", fontsize=12, family="monospace")
    axp = fig.add_axes([0.07, 0.07, 0.88, 0.15], facecolor="#171d23")
    axp.plot(rmse_abl, color="#6b7682", lw=1.0, label="no-message ablation")
    axp.plot(rmse, color="#ffb454", lw=1.6, label="full model")
    axp.set_xlim(0, len(rmse) - 1)
    axp.set_ylim(0, max(rmse.max(), rmse_abl.max()) * 1.1)
    axp.set_xlabel("rollout tick (1 tick = 1 ms)", color="#9aa5b1")
    axp.set_ylabel("voltage RMSE (mV)", color="#9aa5b1")
    axp.tick_params(colors="#9aa5b1")
    axp.legend(loc="lower right", facecolor="#171d23", labelcolor="#c6cdd5", fontsize=8)
    cursor = axp.axvline(0, color="#e8edf2", lw=1)

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1280x720", "-r", str(a.fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(len(truth)):
        for i, m, sc in scatters:
            data = truth if i == 0 else pred
            spk = data[t, :, 3][m] > 0.5
            sc.set_array(np.clip(data[t, :, 0][m], V_LO, V_HI))
            sc.set_sizes(np.where(spk, base_size * 3.2, base_size))
            sc.set_edgecolors(np.where(spk[:, None], np.array([[1, 1, 1, 1.0]]), np.array([[0.16, 0.19, 0.22, 1.0]])))
        cursor.set_xdata([t, t])
        readout.set_text(f"tick {t + 1:3d}/{len(truth)}   voltage RMSE {rmse[t]:5.1f} mV (mean so far {rmse[: t + 1].mean():5.1f})\n"
                         f"spikes this tick: teacher {int(ts[t].sum()):2d}, model {int(ps[t].sum()):2d}   "
                         f"spike overlap F-score so far {running_fscore(ts, ps, t):.2f}")
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", a.out, "frames", len(truth))


if __name__ == "__main__":
    main()
