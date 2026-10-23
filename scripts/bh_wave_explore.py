"""Space-time radial density maps rho(r, t) for several arms, one PNG. Usage: python scripts/bh_wave_explore.py out.png dir tag arm1 arm2 ... [ref=<npz>]"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RLIN = np.arange(0, 48.01, 0.25)


def load(path):
    d = np.load(path)
    return d["rad_lin"].astype(float)


def rho(counts):
    rm = 0.5 * (RLIN[1:] + RLIN[:-1])
    area = np.pi * (RLIN[1:] ** 2 - RLIN[:-1] ** 2)
    n = counts.sum(1, keepdims=True).clip(min=1)
    return counts / area / n


def main():
    out, d, tag = sys.argv[1:4]
    arms = sys.argv[4:]
    fig, ax = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.2), squeeze=False)
    for a, name in zip(ax[0], arms):
        path = name[4:] if name.startswith("ref=") else f"{d}/{tag}_{name}.npz"
        c = load(path)
        r = rho(c)
        a.imshow(np.log10(r.T + 1e-6), origin="lower", aspect="auto", extent=[0, c.shape[0] - 1, 0, 48], vmin=-4.5, vmax=-1.0, cmap="viridis")
        a.set_title(name.split("/")[-1][:30])
        a.set_xlabel("tick")
    ax[0][0].set_ylabel("r")
    plt.tight_layout()
    plt.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
