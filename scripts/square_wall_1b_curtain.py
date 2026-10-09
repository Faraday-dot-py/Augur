import json
import numpy as np

P = np.load("results/scatter_bh_1b_snaps.npz")["pos"]
c = (np.abs(P[0]) == 29.0).any(1)
t = 14
x, y = P[t][:, 0], P[t][:, 1]
res = {}
reg = {"vertical_curtain |x| in [16,21], |y|<12": (np.abs(x) >= 16) & (np.abs(x) < 21) & (np.abs(y) < 12),
       "horizontal curtain |y| in [15,21], |x|<12": (np.abs(y) >= 15) & (np.abs(y) < 21) & (np.abs(x) < 12),
       "|x| or |y| > 21 (outside the curtains)": (np.abs(x) >= 21) | (np.abs(y) >= 21),
       "all": np.ones(len(x), bool)}
for k, m in reg.items():
    res[k] = {"n": int(m.sum()), "n_clipped": int((m & c).sum()), "frac_clipped": float((m & c).sum() / max(m.sum(), 1))}
res["overall_clipped_frac"] = float(c.mean())
res["clipped_with_|x|>=16 or |y|>=15 at t14"] = float(((np.abs(x[c]) >= 16) | (np.abs(y[c]) >= 15)).mean())
res["nonclipped_with_|x|>=16 or |y|>=15 at t14"] = float(((np.abs(x[~c]) >= 16) | (np.abs(y[~c]) >= 15)).mean())
res["nonclipped_n_at_r>15_t14"] = int((np.linalg.norm(P[t][~c], axis=1) > 15).sum())
res["clipped_n_at_r>15_t14"] = int((np.linalg.norm(P[t][c], axis=1) > 15).sum())
json.dump(res, open("results/square_wall/square_wall_1b_curtain.json", "w"), indent=1)
print(json.dumps(res, indent=1))
