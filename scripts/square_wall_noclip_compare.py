import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = np.load("results/scatter_bh_1b_snaps.npz")["pos"][:31]
B = np.load("/home/awebb/polaris-mcp-files/scatter_bh_1b_noclip_snaps.npz")["pos"]
print(A.shape, B.shape, "t0 absmax", np.abs(B[0]).max(), "n at exactly 29:", int((np.abs(B[0]) == 29.0).any(1).sum()), "vs clip run", int((np.abs(A[0]) == 29.0).any(1).sum()))
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
out = {"n_on_clip_line_noclip_run": int((np.abs(B[0]) == 29.0).any(1).sum()), "n_on_clip_line_clip_run": int((np.abs(A[0]) == 29.0).any(1).sum())}
for j, t in enumerate((5, 10, 14, 20)):
    for i, (nm, P) in enumerate((("clipped IC (job 3434)", A), ("resampled IC (job 3447)", B))):
        h, _, _ = np.histogram2d(P[t][:, 0], P[t][:, 1], bins=256, range=[[-32, 32], [-32, 32]])
        ax[i, j].imshow(np.log1p(h).T, origin="lower", cmap="viridis", extent=[-32, 32, -32, 32]); ax[i, j].set_title(f"1e9 subsample {nm} t{t}", fontsize=8)
    for nm, P in (("clip", A), ("resample", B)):
        r = P[t]
        out[f"{nm}_t{t}_n_|x|in[16,21]_|y|<12"] = int(((np.abs(r[:, 0]) >= 16) & (np.abs(r[:, 0]) < 21) & (np.abs(r[:, 1]) < 12)).sum())
        out[f"{nm}_t{t}_n_r>15"] = int((np.linalg.norm(r, axis=1) > 15).sum())
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_noclip.png", dpi=65)
json.dump(out, open("results/square_wall/square_wall_noclip.json", "w"), indent=1)
print(json.dumps(out, indent=1))
