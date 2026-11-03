"""Liquid-argon-style (Lennard-Jones) video: truth vs several scatter-field LJ models on the same initial state.
Inputs are lj_sim / scripts/scatter_lj_eval.py npz files (pos, vel, box, stats with columns step, T, ., E/N, P). Model panels are coloured by
position error against truth (periodic minimum image, in LJ sigma); the truth panel by speed. Bottom: err@k, temperature T, energy per particle E/N.
Usage: python3 scripts/render_lj_generations_video.py --truth results/lj_solid.npz --models label=path.npz ... --title T --out videos/x.mp4"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    tr = np.load(a.truth)
    ms = []
    for s in a.models:
        lab, _, p = s.partition("=")
        ms.append((lab, np.load(p)))
    box = tr["box"]
    nf = min([len(tr["pos"])] + [len(z["pos"]) for _, z in ms])

    def err(P, Q):
        d = P - Q
        d -= box * np.round(d / box)
        return np.linalg.norm(d, axis=-1)

    errs = {lab: np.stack([err(z["pos"][k], tr["pos"][k]) for k in range(nf)]) for lab, z in ms}
    vmax = np.percentile(np.linalg.norm(tr["vel"][:nf:5], axis=-1), 99)
    npn = len(ms) + 1
    fig = plt.figure(figsize=(5.2 * npn, 9.2), dpi=90)
    gs = fig.add_gridspec(2, npn, height_ratios=[1.5, 1])
    cols, txts = [], []
    allp = [("ground truth (exact Lennard-Jones MD)", tr)] + ms
    n = tr["pos"].shape[1]
    for j, (lab, z) in enumerate(allp):
        ax = fig.add_subplot(gs[0, j])
        ax.set_xlim(0, box[0]), ax.set_ylim(0, box[1]), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(lab, fontsize=9)
        kw = dict(cmap="turbo", clim=(0, vmax)) if j == 0 else dict(cmap="viridis", clim=(0, a.vmax))
        col = EllipseCollection(np.full(n, 1.0), np.full(n, 1.0), np.zeros(n), units="xy", offsets=z["pos"][0], offset_transform=ax.transData, **kw)
        ax.add_collection(col)
        cols.append(col)
        txts.append(ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, color="w", bbox=dict(fc="k", alpha=0.6, lw=0)))
    a1, a2, a3 = fig.add_subplot(gs[1, :1]), fig.add_subplot(gs[1, 1:2]), fig.add_subplot(gs[1, 2:])
    ks = np.arange(nf)
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    a2.plot(ks, tr["stats"][:nf, 1], "k:", label="ground truth")
    a3.plot(ks, tr["stats"][:nf, 3], "k:", label="ground truth")
    for j, (lab, z) in enumerate(ms):
        a1.semilogy(ks, np.maximum(errs[lab].mean(1), 1e-5), color=c[j], label=lab)
        a2.plot(ks, z["stats"][:nf, 1], color=c[j])
        a3.plot(ks, z["stats"][:nf, 3], color=c[j])
    a1.set_title("err@k: mean position error per particle (sigma, log)", fontsize=9)
    a2.set_title("temperature T (LJ units)", fontsize=9)
    a3.set_title("energy per particle E/N (LJ units)", fontsize=9)
    a1.legend(fontsize=7, loc="lower right")
    a2.legend(fontsize=7)
    for ax in (a1, a2, a3):
        ax.set_xlabel("frame"), ax.grid(alpha=0.3)
    cur = [ax.axvline(0, color="k", lw=0.8) for ax in (a1, a2, a3)]
    fig.suptitle(f"{a.title}   (N={n}, same initial state; viridis = position error, turbo = speed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    w = animation.FFMpegWriter(fps=a.fps, bitrate=3500)
    with w.saving(fig, a.out, 90):
        for k in range(nf):
            cols[0].set_offsets(tr["pos"][k]), cols[0].set_array(np.linalg.norm(tr["vel"][k], axis=-1))
            txts[0].set_text(f"frame {k}  T {tr['stats'][k, 1]:.3f}  E/N {tr['stats'][k, 3]:.3f}")
            for j, (lab, z) in enumerate(ms, 1):
                cols[j].set_offsets(z["pos"][k]), cols[j].set_array(errs[lab][k])
                txts[j].set_text(f"frame {k}  err {errs[lab][k].mean():.4f}  T {z['stats'][k, 1]:.3f}  E/N {z['stats'][k, 3]:.3f}")
            for cc in cur:
                cc.set_xdata([k, k])
            w.grab_frame()
        for _ in range(a.fps):
            w.grab_frame()
    for lab, _ in ms:
        print(lab, {k: round(float(errs[lab][k].mean()), 4) for k in (1, 5, 10, 20, 50, 100) if k < nf})


if __name__ == "__main__":
    main()
