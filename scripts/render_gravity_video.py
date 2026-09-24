"""Renders the gravity generality test model (checkpoints/gravity_dynamics_v1.pt)
free-running from t=0 against the ground-truth N-body sim, 3 scenes side by side.
Model rollouts beyond step 20 are outside the training horizon.

Usage: PYTHONPATH=. python3 scripts/render_gravity_video.py videos/gravity_test.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs
from scripts.train_gravity_dynamics import unroll

STEPS, TRAIN_H, DT, EPS = 60, 20, 0.1, 0.5
rng = np.random.default_rng(9000)
scenes = []
for n in (3, 5, 8):
    p, v = gs.init_bodies(n, rng)
    scenes.append(gs.rollout(p, v, STEPS, dt=DT, eps=EPS))

dyn = TokenFreeDynamics(n=1000, neighbor_radius=100.0, pair_impulse=True)
dyn.load_state_dict(torch.load("checkpoints/gravity_dynamics_v1.pt"))
model = []
with torch.no_grad():
    for P, V in scenes:
        ps, _ = unroll(dyn, torch.tensor(P[0], dtype=torch.float32), torch.tensor(V[0], dtype=torch.float32), STEPS, DT)
        model.append(np.concatenate([P[:1], ps.numpy()]))

fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), dpi=100)
arts = []
for ax, (P, V), M in zip(axes, scenes, model):
    both = np.concatenate([P, M])
    lo, hi = both.min((0, 1)), both.max((0, 1))
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.1 + 0.5
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    colors = plt.cm.tab10(np.arange(P.shape[1]))
    truth = ax.scatter(P[0, :, 0], P[0, :, 1], s=140, facecolors="none", edgecolors=colors, linewidths=1.5, label="truth")
    mod = ax.scatter(M[0, :, 0], M[0, :, 1], s=30, c=colors, label="model")
    ax.legend(loc="upper right")
    arts.append((truth, mod, ax, P, M))
    ax.set_title(f"{P.shape[1]} bodies")
title = fig.suptitle("")
fig.tight_layout()


def update(i):
    for truth, mod, ax, P, M in arts:
        truth.set_offsets(P[i]); mod.set_offsets(M[i])
        ax.set_xlabel(f"mean pos err {np.linalg.norm(M[i] - P[i], axis=1).mean():.4f}")
    title.set_text(f"N-body gravity, unmodified TokenFreeDynamics: step {i}" + ("  (beyond training horizon)" if i > TRAIN_H else ""))


ani = animation.FuncAnimation(fig, update, frames=STEPS + 1)
ani.save(sys.argv[1], writer=animation.FFMpegWriter(fps=8, bitrate=3000))
