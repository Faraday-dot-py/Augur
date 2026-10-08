"""Render scatter_joint_snaps.py output: snapshot grid PNG (rows truth + models, columns ticks), conserved-quantity series PNG, side-by-side MP4 (streamed frame by frame).
Usage: python3 scripts/render_scatter_joint.py results/scatter_joint/snaps_bh300.npz videos/scatter_joint_bh300 [half]"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
out = sys.argv[2]
half = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
names = ["truth"] + sorted(k[4:] for k in d.files if k.startswith("pos_") and k != "pos_truth")
T = d["pos_truth"].shape[0]
cols = [0, 2, 10, 30, 60, T - 1]
fig, ax = plt.subplots(len(names), len(cols), figsize=(2.6 * len(cols), 2.6 * len(names)))
for r, nm in enumerate(names):
    for c, t in enumerate(cols):
        a = ax[r, c]
        p = d[f"pos_{nm}"][t]
        a.scatter(p[:, 0], p[:, 1], s=2, c="k")
        a.set_xlim(-half, half)
        a.set_ylim(-half, half)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        if r == 0:
            a.set_title(f"tick {t * 10}", fontsize=8)
        if c == 0:
            a.set_ylabel(nm, fontsize=8)
plt.tight_layout()
plt.savefig(out + "_grid.png", dpi=90)
plt.close(fig)

fig, ax = plt.subplots(1, len(("E", "vir", "bound", "rhalf", "outgrid")), figsize=(18, 3.2))
for a, k in zip(ax, ("E", "vir", "bound", "rhalf", "outgrid")):
    for nm in names:
        y = d[f"series_{nm}_{k}"]
        a.plot(y, label=nm, lw=1, ls="--" if nm == "truth" else "-")
    a.set_title(k)
    a.set_xlabel("tick")
    if k in ("vir", "rhalf"):
        a.set_yscale("log")
ax[0].legend(fontsize=6)
plt.tight_layout()
plt.savefig(out + "_series.png", dpi=90)
plt.close(fig)

fig, axs = plt.subplots(1, len(names), figsize=(2.8 * len(names), 3.2), dpi=80)
sc = []
for a, nm in zip(axs, names):
    a.set_xlim(-half, half)
    a.set_ylim(-half, half)
    a.set_aspect("equal")
    a.set_title(nm, fontsize=8)
    a.set_xticks([])
    a.set_yticks([])
    sc.append(a.scatter([], [], s=1.5, c="k"))
txt = fig.suptitle("")
w = animation.FFMpegWriter(fps=12)
with w.saving(fig, out + ".mp4", dpi=80):
    for t in range(T):
        for s, nm in zip(sc, names):
            s.set_offsets(d[f"pos_{nm}"][t])
        txt.set_text(f"tick {t * 10}")
        w.grab_frame()
print("wrote", out)
