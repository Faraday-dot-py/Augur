import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = np.load("results/scatter_bh_1b_snaps.npz")["pos"]
Q = np.load("/home/awebb/polaris-mcp-files/scatter_bh_1b_noclip_snaps.npz")["pos"]
out = {}
for nm, S in (("clip_run_100ticks", P), ("resample_run_30ticks", Q)):
    T = S.shape[0]
    rows = []
    for t in range(T):
        a = np.abs(S[t]).max(1)
        rows.append([t, int(((a >= 29) & (a < 30)).sum()), int(((a >= 30) & (a < 32)).sum()), int((a >= 32).sum())])
    out[nm] = {"t,n[29,30),n[30,32),n[>=32] (of 1e6 bodies; max-norm coordinate)": rows[::5]}
    ids = np.where((np.abs(S[1:6]).max(2) >= 30).any(0))[0]
    out[nm]["n_bodies_with_maxcoord>=30_in_ticks_1..5"] = int(len(ids))
# stationary trap: bodies in [30.5,31.8] at tick t: displacement per tick
S = P
V = np.abs(S[1:]-S[:-1]).max(2) / 0.1
a = np.abs(S[:-1]).max(2)
sel = (a >= 30.5) & (a < 31.8)
out["clip_run_mean_max_axis_speed_of_bodies_in_[30.5,31.8)"] = float(V[sel].mean()) if sel.any() else None
out["clip_run_frac_of_them_with_max_axis_speed<2"] = float((V[sel] < 2).mean())
out["clip_run_frac_all_bodies_axis_speed<2"] = float((V < 2).mean())
# dwell: bodies in [30,32) coordinate at tick 20; where at ticks 40, 60, 100
t0 = 20
a0 = np.abs(S[t0]).max(1)
idx = np.where((a0 >= 30) & (a0 < 32))[0]
out["bodies_in_[30,32)_at_t20"] = int(len(idx))
out["their_maxcoord_pct_at_t[20,40,60,100]"] = {int(t): np.percentile(np.abs(S[t][idx]).max(1), [10, 50, 90]).round(1).tolist() for t in (20, 40, 60, 100)} if len(idx) else None
json.dump(out, open("results/square_wall/square_wall_trap.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if "t," not in k}, indent=1)[:1800])
for nm in ("clip_run_100ticks", "resample_run_30ticks"):
    print(nm, out[nm]["t,n[29,30),n[30,32),n[>=32] (of 1e6 bodies; max-norm coordinate)"])
