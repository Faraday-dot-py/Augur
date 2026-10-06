"""Animated delete-the-star video from scripts/delete_star_trace.py output: inward acceleration vs distance, model vs truth, per step.

Usage: PYTHONPATH=. python3 scripts/render_delete_star.py results/delete_star_E.npz videos/out.mp4 [fps]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
r, warm, ms = d["dists"], int(d["warm"]), float(d["mstar"])
dele, truth = d["delete"], d["truth_before"]
T = dele.shape[1]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=100, gridspec_kw={"width_ratios": [1.3, 1]})
ax.set_xlabel("distance from star")
ax.set_ylabel("inward acceleration")
ax.set_xlim(0, r.max() + 1)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_ylim(-1e-2, truth.max() * 2)
ax.plot(r, truth, "k--", lw=1, label="truth before deletion")
ln, = ax.plot([], [], "o-", c="C3", label="model")
tt, = ax.plot([], [], "k-", lw=2, label="truth now")
ax.legend(loc="upper right")
ttl = ax.set_title("")
ax2.set_xlabel("step")
ax2.set_ylabel("inward acceleration / truth before")
ax2.axvline(warm - 0.5, c="k", ls="--")
ax2.set_xlim(-0.5, T - 0.5)
ax2.set_ylim(-0.2, 1.3)
sel = [i for i in (1, 5, 10, 15, 20) if i < len(r)]
lines = [ax2.plot([], [], c=f"C{k}", label=f"d={r[i]:.0f}")[0] for k, i in enumerate(sel)]
ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97))
w = animation.FFMpegWriter(fps=fps)
with w.saving(fig, sys.argv[2], dpi=100):
    for t in range(T):
        ln.set_data(r, dele[:, t])
        tt.set_data(r, truth if t < warm else np.zeros_like(truth))
        ttl.set_text(f"step {t}: " + ("star present" if t < warm else f"star deleted (+{t - warm})"))
        for l, i in zip(lines, sel):
            l.set_data(np.arange(t + 1), dele[i, :t + 1] / truth[i])
        w.grab_frame()
    for _ in range(fps):
        w.grab_frame()
