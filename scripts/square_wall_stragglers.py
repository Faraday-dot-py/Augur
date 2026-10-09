import numpy as np
z = np.load("results/scatter_bh_100000_snaps.npz")
M, T = z["pos"], z["truth_pos"]
for t in (57, 71, 86, 100):
    rm, rt = np.linalg.norm(M[t], axis=1), np.linalg.norm(T[t], axis=1)
    cut = np.percentile(rt, 99.9)
    print(t, "truth r99.9", round(float(cut), 1), "n model beyond truth r99.9:", int((rm > cut).sum()), "n truth beyond:", int((rt > cut).sum()))
