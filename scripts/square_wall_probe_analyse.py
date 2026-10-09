import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lines = open(sys.argv[1]).read().splitlines()
print(lines[0])
stat = [json.loads(l[7:]) for l in lines if l.startswith("STATIC ")]
roll = [json.loads(l[5:]) for l in lines if l.startswith("ROLL ")][0]
out = {}
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for j, s in enumerate(stat):
    xs = np.array(s["xs"]); k = len(xs)
    ex = np.array(s["exact"]); fu = np.array(s["full"]); me = np.array(s["mesh"])
    o = {}
    for d, nm in enumerate(("+x axis", "+y axis", "diag")):
        sl = slice(d * k, (d + 1) * k)
        ux = np.array([1, 0]) if d == 0 else np.array([0, 1]) if d == 1 else np.array([1, 1]) / np.sqrt(2)
        r = xs * (1 if d < 2 else 1)
        rad_ex = ex[sl] @ ux; rad_fu = fu[sl] @ ux; rad_me = me[sl] @ ux
        o[nm] = {"r": r.tolist(), "radial_exact": rad_ex.tolist(), "radial_full": rad_fu.tolist(), "radial_mesh": rad_me.tolist()}
        if d == 0:
            ax[0, j].plot(r, rad_ex, "k", label="exact"); ax[0, j].plot(r, rad_fu, "r", label="model full"); ax[0, j].plot(r, rad_me, "b--", label="model mesh only")
            ax[0, j].axvline(32, c="gray"); ax[0, j].axvline(29, c="gray", ls=":"); ax[0, j].set_title(f"{s['kind']} n={s['n']}: radial accel on +x axis tracers"); ax[0, j].legend(fontsize=7)
            ax[1, j].plot(r, rad_fu / rad_ex, "r"); ax[1, j].plot(r, rad_me / rad_ex, "b--"); ax[1, j].axvline(32, c="gray"); ax[1, j].set_ylim(-2, 3); ax[1, j].set_title("ratio model/exact")
    out[s["kind"]] = o
    near = (xs >= 28) & (xs < 33)
    print(s["kind"], "x:", [(float(x), round(float(a), 1), round(float(b), 1), round(float(m), 1)) for x, a, b, m in zip(xs[::2], ex[:k, 0][::2], fu[:k, 0][::2], me[:k, 0][::2])])
json.dump(out, open("results/square_wall/square_wall_probe_static.json", "w"))
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_probe_static.png", dpi=65)

mr, tr = np.array(roll["model_r"]), np.array(roll["truth_r"])
res = {"model_rmax_final_median": float(np.median(mr[-1])), "truth_rmax_final_median": float(np.median(tr[-1]))}
for nm, R in (("model", mr), ("truth", tr)):
    res[nm] = {"max_r_over_run_median": float(np.median(R.max(0))), "frac_crossed_32": float((R.max(0) > 32).mean()), "frac_reach_45": float((R.max(0) > 45).mean()),
               "final_r_pct10_50_90": np.percentile(R[-1], [10, 50, 90]).round(1).tolist(),
               "frac_turned_back_after_crossing": float(((R.max(0) > 32) & (R[-1] < R.max(0) - 3)).sum() / max(1, (R.max(0) > 32).sum()))}
res["radius_traj_median"] = {"t": [0, 5, 10, 20, 40, 60, 100], "model": [float(np.median(mr[t])) for t in (0, 5, 10, 20, 40, 60, 100)], "truth": [float(np.median(tr[t])) for t in (0, 5, 10, 20, 40, 60, 100)]}
json.dump(res, open("results/square_wall/square_wall_probe_rollout.json", "w"), indent=1)
print(json.dumps(res, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for R, a, nm in ((tr, ax[0], "truth"), (mr, ax[1], "model")):
    a.plot(R[:, :60], lw=.4); a.axhline(32, c="k"); a.set_title(f"{nm}: massless tracers launched outward at 0.95c from r~24, N=2e4 cluster"); a.set_xlabel("tick"); a.set_ylabel("r")
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_probe_rollout.png", dpi=65)
