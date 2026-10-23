"""Radial density profiles at fixed ticks, overlaid across arms (and N). Usage: python scripts/bh_wave_profiles.py <data_dir> <ref_dir> <out_png> <scene> <ticks comma> [ref npz stem]"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

RLIN = np.arange(0, 48.01, 0.25)
RMID = 0.5 * (RLIN[1:] + RLIN[:-1])
AREA = np.pi * (RLIN[1:] ** 2 - RLIN[:-1] ** 2)


def prof(c, t, sm=0):
    f = c[t] / c[t].sum() / AREA
    return gaussian_filter1d(f, sm) if sm else f


def main():
    d, rd, out, scene, ticks = sys.argv[1:6]
    ref = sys.argv[6] if len(sys.argv) > 6 else None
    ticks = [int(t) for t in ticks.split(",")]
    arms = [("truth4", "k"), ("pm128", "tab:blue"), ("pm128s4", "tab:cyan"), ("model_new", "tab:red")]
    fig, ax = plt.subplots(1, len(ticks), figsize=(4.6 * len(ticks), 4.2), squeeze=False)
    for a, t in zip(ax[0], ticks):
        for name, col in arms:
            try:
                c = np.load(f"{d}/{scene}_{name}.npz")["rad_lin"].astype(float)
            except FileNotFoundError:
                continue
            a.semilogy(RMID, prof(c, t), color=col, lw=1.0, label=name)
        if ref:
            c = np.load(f"{rd}/{ref}.npz")["rad_lin"].astype(float)
            a.semilogy(RMID, prof(c, t), color="tab:orange", lw=1.0, label="axisym ref")
        a.set_title(f"tick {t}")
        a.set_xlim(0, 30)
        a.set_ylim(1e-6, 1)
        a.set_xlabel("r")
    ax[0][0].set_ylabel("surface density / N")
    ax[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=75)


if __name__ == "__main__":
    main()
