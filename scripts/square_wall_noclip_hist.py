import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

A = np.load("results/scatter_bh_1b_hist.npz")["full"]
B = np.load("/home/awebb/polaris-mcp-files/scatter_bh_1b_noclip_hist.npz")["full"]
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for j, t in enumerate((0, 5, 14, 20)):
    for i, H in enumerate((A, B)):
        f = H[t].astype(float)
        ax[i, j].imshow(np.maximum(f, .5).T, origin="lower", cmap="viridis", norm=LogNorm(vmin=1, vmax=max(2, f.max())))
        ax[i, j].set_title(("clipped IC (3434) " if i == 0 else "resampled IC (3447) ") + f"full 1e9 hist t{t} (box +-32)", fontsize=8)
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_noclip_hist.png", dpi=65)
