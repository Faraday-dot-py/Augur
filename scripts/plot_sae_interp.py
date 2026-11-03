"""Render diagnostic plots from an sae_interp npz. Usage: python scripts/plot_sae_interp.py results/sae_interp_topk_dec0_s4738.npz videos/sae_interp_topk_dec0_s4738"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAWS = ["pow1", "pow1.5", "pow2", "pow3", "yuk8", "yuk32", "yuk128", "lj"]
z = np.load(sys.argv[1])
prefix = sys.argv[2]
for part, cont in (("mesh", "log_cell_mass"), ("pair", "log_r")):
    sizes = sorted({int(k.split("_")[1]) for k in z.files if k.startswith(part + "_") and k.endswith("_law_eta2")})
    if not sizes:
        continue
    fig, ax = plt.subplots(2, len(sizes), figsize=(5 * len(sizes), 8), squeeze=False)
    for j, s in enumerate(sizes):
        e, c = z[f"{part}_{s}_law_eta2"], z[f"{part}_{s}_{cont}_abscorr"]
        codes, lab, y = z[f"{part}_{s}_heldout_codes"].astype(np.float32), z[f"{part}_{s}_heldout_law"], z[f"{part}_{s}_heldout_{cont}"]
        live = (codes > 0).any(0)
        ax[0, j].scatter(e[live], c[live], s=6, alpha=0.5)
        ax[0, j].set(xlabel="law eta^2 (per live feature)", ylabel=f"|corr| with {cont}", title=f"{part} dictionary {s}: {int(live.sum())} live of {len(e)}")
        top = np.argsort(e * live)[::-1][:4]
        for f in top:
            ax[1, j].scatter(y, codes[:, f], s=3, alpha=0.4, label=f"feature {f} (eta^2 {e[f]:.2f}, mostly {LAWS[int(np.argmax([codes[lab == i, f].mean() for i in range(8)]))]})")
        ax[1, j].set(xlabel=cont, ylabel="activation (held-out)")
        ax[1, j].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{prefix}_{part}.png", dpi=110)
    print("wrote", f"{prefix}_{part}.png")
