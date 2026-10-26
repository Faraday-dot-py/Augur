"""Side-by-side video: ground truth vs several model generations on one shared scene.
Top row: bodies per panel (model panels coloured by position error against truth; seams drawn for tiled panels). Bottom row:
err@k (mean position error per body at tick k, sim units), relative energy drift |E(k)-E(0)|/|E(0)|, and err@k split by tick-0 local density (dense third solid, sparse third dashed).
Frames are streamed to ffmpeg one at a time.
Usage: python3 scripts/render_generations_video.py --npz a.npz [b.npz ...] --panels key=label[:tile32] ... --title T --out videos/x.mp4"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--panels", nargs="+", required=True, help="key=label or key=label@seam<tile>; key is the npz prefix")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--view", type=float, default=0.0, help="half-width of the plotted window (default span/2)")
    ap.add_argument("--vmax", type=float, default=2.0)
    a = ap.parse_args()
    d = {}
    for f in a.npz:
        d.update(dict(np.load(f)))
    span = float(d["span"])
    tp = d["truth_pos"]
    T = tp.shape[0]
    n = tp.shape[1]
    panels = []
    for s in a.panels:
        key, _, lab = s.partition("=")
        lab, _, seam = lab.partition("@")
        panels.append((key, lab, float(seam) if seam else None))
    mdl = [p for p in panels if p[0] != "truth"]
    d0 = np.linalg.norm(tp[0][:, None] - tp[0][None], axis=-1)
    dens = (d0 < 3.0).sum(1)
    lo, hi = np.quantile(dens, [1 / 3, 2 / 3])
    dense, sparse = dens >= hi, dens <= lo
    err = {k: np.linalg.norm(d[f"{k}_pos"] - tp, axis=-1) for k, _, _ in mdl}
    e0 = d["truth_E"][0]
    drift = {k: np.abs(d[f"{k}_E"] - d[f"{k}_E"][0]) / abs(e0) for k, _, _ in mdl}
    drift["truth"] = np.abs(d["truth_E"] - e0) / abs(e0)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    npn = len(panels)
    fig = plt.figure(figsize=(4.4 * npn, 9.2), dpi=90)
    gs = fig.add_gridspec(2, npn + 1, height_ratios=[1.35, 1], width_ratios=[1] * npn + [0.06])
    sc, txt = {}, {}
    half = span / 2
    view = a.view or half
    for i, (key, lab, seam) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        pos = d[f"{key}_pos"] if key != "truth" else tp
        kw = dict(c=dens, cmap="gray_r", vmin=0, vmax=dens.max()) if key == "truth" else dict(c=err[key][0], cmap="viridis", vmin=0, vmax=a.vmax)
        sc[key] = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=6, **kw)
        ax.set_xlim(-view, view), ax.set_ylim(-view, view), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(lab, fontsize=10)
        if seam:
            for g in np.arange(-half + seam, half - 1e-6, seam):
                ax.axvline(g, color="r", lw=0.6, ls=":")
                ax.axhline(g, color="r", lw=0.6, ls=":")
        txt[key] = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, color="w", bbox=dict(fc="k", alpha=0.6, lw=0))
    cb = fig.colorbar(sc[mdl[0][0]], cax=fig.add_subplot(gs[0, npn]))
    cb.set_label("position error vs truth (sim units)")
    ks = np.arange(T)
    a1, a2, a3 = fig.add_subplot(gs[1, :npn // 3 + 1]), fig.add_subplot(gs[1, npn // 3 + 1:2 * npn // 3 + 1]), fig.add_subplot(gs[1, 2 * npn // 3 + 1:npn])
    for j, (k, lab, _) in enumerate(mdl):
        c = cols[j]
        a1.plot(ks, err[k].mean(1), color=c, label=lab)
        a2.plot(ks, drift[k], color=c)
        a3.plot(ks, err[k][:, dense].mean(1), color=c)
        a3.plot(ks, err[k][:, sparse].mean(1), color=c, ls="--")
    a2.plot(ks, drift["truth"], color="k", ls=":", label="ground truth (integrator floor)")
    a1.set_title("err@k: mean position error per body", fontsize=9)
    a2.set_title("relative energy drift |E(k)-E(0)| / |E(0)|", fontsize=9)
    a2.set_yscale("log")
    a3.set_title("err@k, dense third (solid) vs sparse third (dashed)\nof bodies by tick-0 neighbours within 3 units", fontsize=9)
    for ax in (a1, a2, a3):
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
    a1.legend(fontsize=7, loc="upper left")
    a2.legend(fontsize=7, loc="lower right")
    cur = [ax.axvline(0, color="k", lw=0.8) for ax in (a1, a2, a3)]
    fig.suptitle(f"{a.title}   ({n} bodies, span {span:g}, tick 0 = identical initial state)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def draw(k):
        for key, lab, seam in panels:
            pos = d[f"{key}_pos"] if key != "truth" else tp
            sc[key].set_offsets(pos[k])
            if key != "truth":
                sc[key].set_array(err[key][k])
                txt[key].set_text(f"tick {k}  err@{k} {err[key][k].mean():.3f}\ndE/E {drift[key][k]:.4f}")
            else:
                txt[key].set_text(f"tick {k}  dE/E {drift['truth'][k]:.5f}")
        for c in cur:
            c.set_xdata([k, k])

    w = animation.FFMpegWriter(fps=a.fps, bitrate=3500)
    with w.saving(fig, a.out, 90):
        for k in range(T):
            draw(k)
            w.grab_frame()
        for _ in range(a.fps):
            w.grab_frame()
    ks_ = [5, 10, 20, 50, 100]
    for k, lab, _ in mdl:
        print(lab, {t: round(float(err[k][t].mean()), 3) for t in ks_ if t < T}, "dE@100", round(float(drift[k][min(100, T - 1)]), 4),
              "dense/sparse err@50", round(float(err[k][min(50, T - 1)][dense].mean()), 3), round(float(err[k][min(50, T - 1)][sparse].mean()), 3))


if __name__ == "__main__":
    main()
