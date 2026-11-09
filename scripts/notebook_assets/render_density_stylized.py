import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "docs/design/assets/"
ink_brick = LinearSegmentedColormap.from_list("ink_brick", [
    (0.12, 0.11, 0.09, 0.0), (0.12, 0.11, 0.09, 0.85), (0.12, 0.11, 0.09, 1.0), (0.64, 0.23, 0.17, 1.0)],
    N=256)
ink_brick = LinearSegmentedColormap.from_list("ink_brick", [
    (0.0, (0.12, 0.11, 0.09, 0.0)), (0.35, (0.12, 0.11, 0.09, 0.55)), (0.7, (0.12, 0.11, 0.09, 1.0)),
    (0.85, (0.45, 0.17, 0.13, 1.0)), (1.0, (0.64, 0.23, 0.17, 1.0))])


def density(arr, name, gamma=0.5):
    a = np.log1p(arr.astype(np.float64))
    a = (a / a.max()) ** gamma
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(a, cmap=ink_brick, origin="lower", interpolation="bilinear", vmin=0, vmax=1)
    fig.savefig(OUT + name, dpi=200, transparent=True)
    plt.close(fig)


d = np.load("results/bh_1e9_edge_free_ticks.npz")
density(d["full"][-1], "density_1e9_stylized.png")
h = np.load("results/showcase_bh_1e10_kpadext_hist.npz")
density(h["full"][-1], "density_1e10_stylized.png")
