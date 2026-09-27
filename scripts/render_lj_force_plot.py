"""Interpretability figure: the learned radial pair force f(d) (and its integral, the potential) against the
analytic Lennard-Jones curve, over the checkpoints supplied. The point is that f(d) is a single learned
scalar function of distance, so it can be read off and checked against the true physics directly -- no
probing or activation-atlas needed.

Usage: PYTHONPATH=. python3 scripts/render_lj_force_plot.py out.png v2=checkpoints/lj_force_v2.pt v3=checkpoints/lj_force_v3.pt
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.lj_force import LJForceDynamics
from scripts import lj_sim

out = sys.argv[1]
ckpts = [a.split("=", 1) for a in sys.argv[2:]]
d = torch.linspace(0.75, 2.5, 600, dtype=torch.float64)
dn = d.numpy()
_, f_rc = lj_sim._sf_consts(lj_sim.RC)
f_true = 24 * (2 * dn ** -13 - dn ** -7) - f_rc
u_rc, _ = lj_sim._sf_consts(lj_sim.RC)
u_true = 4 * (dn ** -12 - dn ** -6) - u_rc + f_rc * (dn - lj_sim.RC)

fig, (axf, axu, axe) = plt.subplots(1, 3, figsize=(15.5, 4.5))
axf.plot(dn, f_true, "k--", lw=2, label="analytic LJ")
axu.plot(dn, u_true, "k--", lw=2, label="analytic LJ")
axf.axhline(0, color="0.85", lw=1, zorder=0)
axu.axhline(0, color="0.85", lw=1, zorder=0)
for label, path in ckpts:
    ck = torch.load(path, map_location="cpu")
    model = LJForceDynamics(width=ck["args"].get("width", 64)).double()
    model.load_state_dict(ck["state"])
    model.requires_grad_(False)
    model.build_table()
    with torch.no_grad():
        f = model.pair_force(d).numpy()
        u = model.potential(d).numpy()
    axf.plot(dn, f, lw=1.8, label=f"learned ({label})")
    axu.plot(dn, u, lw=1.8, label=f"learned ({label})")
    axe.plot(dn, np.abs(f - f_true), lw=1.8, label=f"|error| ({label})")

# symlog so both the deep repulsive-core blowup (d < ~0.9) and the shallow attractive well
# (d > ~1.5) are visible on one axis -- a plain linear or clipped axis hides the core mismatch.
for ax, ylab, title in ((axf, "f(d)", "pair force"), (axu, "U(d)", "pair potential (integral of f)")):
    ax.set_xlabel("distance d (sigma)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axvline(1.0, color="0.9", lw=8, zorder=0)
    ax.legend(fontsize=9)
axe.set_xlabel("distance d (sigma)")
axe.set_ylabel("|f_learned - f_true|")
axe.set_title("force error (log scale)")
axe.set_yscale("log")
axe.axvline(1.0, color="0.9", lw=8, zorder=0)
axe.axvspan(0.75, 1.2, color="0.93", zorder=-1)
axe.legend(fontsize=9)
fig.suptitle("Learned pair-interaction law vs the true Lennard-Jones 12-6 curve (repulsive core rarely visited in training data)")
fig.tight_layout()
fig.savefig(out, dpi=130)
