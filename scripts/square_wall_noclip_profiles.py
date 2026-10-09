import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = np.load("results/scatter_bh_1b_hist.npz")["full"]
B = np.load("/home/awebb/polaris-mcp-files/scatter_bh_1b_noclip_hist.npz")["full"]
xs = (np.arange(1024) + .5) / 16 - 32
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for j, t in enumerate((0, 5, 10, 14)):
    for i, H in enumerate((A, B)):
        f = H[t].astype(float)
        sel = np.abs(xs) < 10
        ax[0, j].semilogy(xs, f[:, sel].sum(1) + 1e-1, label=("clip", "resample")[i])
        ax[1, j].semilogy(xs, f[sel, :].sum(0) + 1e-1, label=("clip", "resample")[i])
    ax[0, j].set_title(f"t{t}: x-profile (sum over |y|<10; axis0 = x)", fontsize=8); ax[1, j].set_title(f"t{t}: y-profile (sum over |x|<10)", fontsize=8); ax[0, j].legend()
    for a in (ax[0, j], ax[1, j]):
        a.axvline(29, c="gray", ls=":"); a.axvline(-29, c="gray", ls=":")
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_noclip_profiles.png", dpi=65)
