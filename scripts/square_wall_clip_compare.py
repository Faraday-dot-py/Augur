import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("results/scatter_bh_100000_snaps.npz")
M, T = z["pos"], z["truth_pos"]
rng = np.random.default_rng(4738)
p0 = np.clip(rng.normal(0.0, 8.0, (M.shape[1], 2)), -29.0, 29.0)
clip = (np.abs(p0) == 29.0).any(1)
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for j, t in enumerate((5, 10, 14, 20)):
    for i, (nm, P) in enumerate((("truth", T), ("model", M))):
        a = ax[i, j]
        a.scatter(P[t][~clip][::10, 0], P[t][~clip][::10, 1], s=.3, c="lightgray")
        a.scatter(P[t][clip][:, 0], P[t][clip][:, 1], s=5, c="r")
        a.set_xlim(-45, 45); a.set_ylim(-45, 45); a.set_title(f"100k {nm} t{t}: IC-clipped bodies (red, n={clip.sum()})", fontsize=8)
        a.axvline(32, lw=.4); a.axvline(-32, lw=.4); a.axhline(32, lw=.4); a.axhline(-32, lw=.4)
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_clip_100k.png", dpi=65)
s = np.load("results/scatter_bh_1b_snaps.npz")["pos"]
for t in (14, 57, 71, 100):
    r = np.linalg.norm(s[t], axis=1)
    print("1e9 sub t", t, "r pct 50/90/99/99.9", np.percentile(r, [50, 90, 99, 99.9]).round(1), "frac r>32", (r > 32).mean().round(4), "frac outside box", (np.abs(s[t]) > 32).any(1).mean().round(4))
