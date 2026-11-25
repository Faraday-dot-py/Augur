"""Side-by-side contact video: truth vs several contact models on the same scene, from scripts/scatter_contact_scenes.py trajectory npz files
(truth/model/mask/rad/prims; every file must hold the same truth). Bottom: mean position error per body (log) and kinetic-energy ratio model/truth.
Usage: python3 scripts/render_contact_generations_video.py --family disc --scene 0 --models label=path.npz ... --out videos/x.mp4"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle


def style(ax, lim, prims):
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lim", type=float, default=11.0)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    ms = []
    for s in a.models:
        lab, _, p = s.partition("=")
        z = np.load(p, allow_pickle=True)
        ms.append((lab, z))
    z0 = ms[0][1]
    i = a.scene
    for _, z in ms:
        assert np.array_equal(z["truth"], z0["truth"]), "scenes differ"
    prims = json.loads(str(z0["prims"]))[i]
    n = int(z0["mask"][i].sum())
    rad = z0["rad"][i][:n]
    truth = z0["truth"][i][:, :n]
    T = truth.shape[0]
    err = {lab: np.linalg.norm(z["model"][i][:, :n] - truth, axis=-1).mean(1) for lab, z in ms}
    vel = lambda P: np.diff(P, axis=0, prepend=P[:1])
    ke_t = (vel(truth) ** 2).sum((1, 2))
    ke = {lab: (vel(z["model"][i][:, :n]) ** 2).sum((1, 2)) / np.maximum(ke_t, 1e-9) for lab, z in ms}
    panels = [("ground truth", truth)] + [(lab, z["model"][i][:, :n]) for lab, z in ms]
    npn = len(panels)
    fig = plt.figure(figsize=(4.2 * npn, 8.6), dpi=90)
    gs = fig.add_gridspec(2, npn, height_ratios=[1.5, 1])
    patches = []
    for j, (lab, P) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        style(ax, a.lim, prims)
        ax.set_title(lab, fontsize=9)
        cs = [Circle(P[0, k], rad[k], fc="tab:blue" if j == 0 else "tab:red", alpha=0.6, ec="k", lw=0.5) for k in range(n)]
        for c in cs:
            ax.add_patch(c)
        patches.append(cs)
    a1, a2 = fig.add_subplot(gs[1, : npn // 2 + 1]), fig.add_subplot(gs[1, npn // 2 + 1:])
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, (lab, _) in enumerate(ms):
        a1.semilogy(np.maximum(err[lab], 1e-4), color=cols[j], label=lab)
        a2.semilogy(np.maximum(ke[lab], 1e-4), color=cols[j])
    a1.set_title("mean position error per body vs truth (sim units, log)", fontsize=9)
    a2.set_title("kinetic-energy ratio, model / truth (per-tick displacement^2, log; 1 = same)", fontsize=9)
    a2.axhline(1, color="k", lw=0.6, ls=":")
    a1.legend(fontsize=7)
    for ax in (a1, a2):
        ax.set_xlabel("tick"), ax.grid(alpha=0.3)
    cur = [ax.axvline(0, color="k", lw=0.8) for ax in (a1, a2)]
    fig.suptitle(f"{a.title}   (same 15-ball scene; blue = truth, red = model)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    w = animation.FFMpegWriter(fps=a.fps, bitrate=3000)
    with w.saving(fig, a.out, 90):
        for t in range(T):
            for (lab, P), cs in zip(panels, patches):
                for k, c in enumerate(cs):
                    c.center = P[t, k]
            for c in cur:
                c.set_xdata([t, t])
            w.grab_frame()
        for _ in range(a.fps):
            w.grab_frame()
    for lab, _ in ms:
        print(lab, {t: round(float(err[lab][t]), 2) for t in (5, 10, 20, 40, 60, 100) if t < T})


if __name__ == "__main__":
    main()
