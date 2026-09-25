"""Density-frame grid: rows = providers, columns = snapshot steps; per column the extent is fixed by the first row (exact), log colour
scale shared within a column. Usage: python scripts/rollout_render.py --out videos/x.png --steps 0 500 --rows exact=results/a_snaps.npz adaptive=..."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--bins", type=int, default=400)
    args = ap.parse_args()
    names = [r.split("=")[0] for r in args.rows]
    snaps = []
    for r in args.rows:
        z = np.load(r.split("=", 1)[1])
        snaps.append((z["pos"], list(z["steps"])))
    fig, ax = plt.subplots(len(names), len(args.steps), figsize=(3.2 * len(args.steps), 3.2 * len(names)), squeeze=False)
    for j, s in enumerate(args.steps):
        p0 = snaps[0][0][snaps[0][1].index(s)]
        c = np.median(p0, 0)
        half = np.quantile(np.linalg.norm(p0 - c, axis=1), 0.99) * 1.1
        hs = []
        for pos, st in snaps:
            p = pos[st.index(s)]
            hs.append(np.histogram2d(p[:, 0], p[:, 1], bins=args.bins, range=[[c[0] - half, c[0] + half], [c[1] - half, c[1] + half]])[0])
        vmax = np.log1p(max(h.max() for h in hs))
        for i, h in enumerate(hs):
            ax[i, j].imshow(np.log1p(h.T), origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(f"step {s}")
            if j == 0:
                ax[i, j].set_ylabel(names[i])
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)


if __name__ == "__main__":
    main()
