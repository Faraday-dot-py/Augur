"""Usage: PYTHONPATH=. python3 scripts/plot_far_field_accuracy.py results/far_field_accuracy.npz out.png"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load(sys.argv[1])
t = d["step"] * float(d["dt"])
names = [str(d[f"name_{i}"]) for i in range(10) if f"name_{i}" in d]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=90)
for i, name in enumerate(names):
    if i == 0:
        continue
    axes[0].plot(t[1:], d[f"gap_{i}"][1:], label=name)
    axes[1].plot(t, d[f"energy_{i}"] / abs(d["energy_0"][0]) - d["energy_0"] / abs(d["energy_0"][0]), label=name)
    axes[2].plot(t, d[f"mom_{i}"], label=name)
axes[0].set_yscale("log"); axes[0].set_title("mean position gap vs exact truth"); axes[0].set_ylabel("distance")
axes[1].set_title("energy minus truth energy, / |E0|")
axes[2].set_yscale("log"); axes[2].set_title("net momentum |P|")
for ax in axes:
    ax.set_xlabel("t"); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle(f"{int(d['bodies']):,} bodies, uniform scaled init")
fig.tight_layout()
fig.savefig(sys.argv[2])
