"""Render truth vs token-model vs central-model rollouts for the relativistic
gravity learning test (job 3119), side by side per scene, color-coded by
speed/c and flagging bodies over the speed limit.

Usage: PYTHONPATH=. python3 scripts/render_gravity_relativistic_video.py results/gravity_relativistic_rollout.npz videos/out.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
out = sys.argv[2]
c = float(d["c"])
n_scenes = int(d["n_scenes"])
cols = ["truth", "token", "central"]

fig, axes = plt.subplots(n_scenes, 3, figsize=(9, 3 * n_scenes), dpi=90)
if n_scenes == 1:
    axes = axes[None, :]

scenes = []
for i in range(n_scenes):
    s = {k: d[f"{k}_{i}"] for k in ("truth_pos", "truth_vel", "token_pos", "token_vel", "central_pos", "central_vel")}
    scenes.append(s)
    allp = np.concatenate([s["truth_pos"], s["token_pos"], s["central_pos"]], 0)
    lo, hi = allp.min(), allp.max()
    pad = 0.1 * (hi - lo + 1e-6)
    for j in range(3):
        ax = axes[i, j]
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(cols[j])

steps = scenes[0]["truth_pos"].shape[0]
artists = [[None] * 3 for _ in range(n_scenes)]


def draw(t):
    out_artists = []
    for i, s in enumerate(scenes):
        for j, name in enumerate(cols):
            ax = axes[i, j]
            if artists[i][j] is not None:
                artists[i][j].remove()
            pos, vel = s[f"{name}_pos"][t], s[f"{name}_vel"][t]
            speed = np.linalg.norm(vel, axis=-1)
            over = speed > c
            colors = np.where(over, "red", "black")
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=25 + 40 * (speed / c).clip(0, 1))
            artists[i][j] = sc
            out_artists.append(sc)
    fig.suptitle(f"step {t}/{steps - 1}  (red = |v| > c={c})")
    return out_artists


ani = animation.FuncAnimation(fig, draw, frames=steps, interval=250, blit=False)
ani.save(out, writer="ffmpeg", fps=6, dpi=90)
print(f"saved {out}", flush=True)
