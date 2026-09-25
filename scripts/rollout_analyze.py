"""Summarise rollout_nbody.py results: table per scenario + figures. Usage: python scripts/rollout_analyze.py <scenario> ...
Scenarios: flyby (tags rollout_flyby_*), flyby12000 (rollout_s12000_*), uniform (rollout_uni_*). Reads results/<tag>.json."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PREFIX = {"flyby": "rollout_flyby_", "flyby12000": "rollout_s12000_", "uniform": "rollout_uni_"}
ORDER = ["exact", "adaptive", "geo_audit", "est", "geo", "bh", "mesh1024", "mesh2048"]
QS = (0.25, 0.5, 0.75, 0.9, 0.99)
OUT = "videos/rollout_figs"


def load(scn):
    d = {}
    for p in ORDER:
        f = f"results/{PREFIX[scn]}{p}.json"
        if os.path.exists(f):
            d[p] = json.load(open(f))
    return d


def series(r, key):
    return np.array([x["step"] for x in r["diags"]]), np.array([x[key] for x in r["diags"]])


def summarize(scn, d, floor):
    ex = d["exact"]
    st, E = series(ex, "E")
    E0 = abs(E[0])
    rows = []
    for p, r in d.items():
        s, e = series(r, "E")
        _, P = series(r, "P")
        _, vs = series(r, "vsum")
        _, L = series(r, "L")
        com = np.array([x["com"] for x in r["diags"]])
        n = min(len(e), len(E))
        row = {"provider": p, "steps": int(s[-1]), "dE_over_E0_final": float((e[n - 1] - e[0]) / E0), "dE_max": float(np.abs((e[:n] - e[0]) / E0).max()),
               "dE_vs_exact_final": float((e[n - 1] - E[n - 1]) / E0), "P_rel_max": float((P / vs).max()), "L_drift_rel_final": float((L[-1] - L[0]) / abs(L[0])),
               "L_drift_rel_max": float(np.abs((L - L[0]) / abs(L[0])).max()), "com_drift_max": float(np.linalg.norm(com - com[0], axis=1).max()),
               "step_s": r["step_time_mean"], "wall_s": r["wall_s"]}
        f = r.get("force")
        if f:
            row.update({"cost": f["mean_cost"], "force_s": f["mean_force_time_s"], "audit_s_per_audit": f["mean_audit_time_s"],
                        "audit_mean": float(np.mean([a[2] for a in f["audits"]])) if f["audits"] else None,
                        "audit_max": float(np.max([a[2] for a in f["audits"]])) if f["audits"] else None,
                        "audit_frac_over_target": float(np.mean([a[2] > 0.01 for a in f["audits"]])) if f["audits"] else None,
                        "events": [e for e in f["events"] if e["event"] == "fallback"], "n_probe": sum(e["event"] != "fallback" for e in f["events"]),
                        "est_frac": f["mode_fraction_est"]})
        v = r.get("vs_ref")
        if v and p != "exact":
            row["vs_ref"] = {x["step"]: x for x in v}
        rows.append(row)
    return rows


def plots(scn, d, floor):
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ex = d["exact"]
    st, E = series(ex, "E")
    for p, r in d.items():
        s, e = series(r, "E")
        _, L = series(r, "L")
        ax[0, 0].plot(s, (e - e[0]) / abs(E[0]), label=p)
        ax[0, 1].plot(s, (L - L[0]) / abs(L[0]), label=p)
        _, P = series(r, "P")
        _, vs = series(r, "vsum")
        ax[0, 2].semilogy(s, np.maximum(P / vs, 1e-20), label=p)
        v = r.get("vs_ref")
        if v and p != "exact":
            ss = [x["step"] for x in v]
            ax[1, 0].plot(ss, [x["rq_ratio"][2] for x in v], label=p)
            ax[1, 1].plot(ss, [x["rq_ratio"][4] for x in v], label=p)
            ax[1, 2].plot(ss, [x["density_corr"] for x in v], label=p)
    if floor is not None:
        v = floor["vs_ref"]
        ss = [x["step"] for x in v]
        ax[1, 0].plot(ss, [x["rq_ratio"][2] for x in v], "k--", label="noise floor (1e-6 perturbed exact)")
        ax[1, 1].plot(ss, [x["rq_ratio"][4] for x in v], "k--")
        ax[1, 2].plot(ss, [x["density_corr"] for x in v], "k--")
    for a, t in zip(ax.ravel(), ["(E-E0)/|E0|", "(L-L0)/|L0|", "|P| / sum|v|", "r75 ratio vs exact", "r99 ratio vs exact", "density corr vs exact"]):
        a.set_title(t)
        a.set_xlabel("step")
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_yscale("symlog", linthresh=1e-6)
    ax[0, 1].set_yscale("symlog", linthresh=1e-6)
    fig.suptitle(scn)
    fig.tight_layout()
    f = f"{OUT}/rollout_{scn}_conservation_stats.png"
    fig.savefig(f, dpi=110)
    return f


def main():
    for scn in sys.argv[1:]:
        d = load(scn)
        floor = json.load(open("results/rollout_flyby_perturb.json")) if scn == "flyby" and os.path.exists("results/rollout_flyby_perturb.json") else None
        rows = summarize(scn, d, floor)
        json.dump(rows, open(f"results/rollout_summary_{scn}.json", "w"), indent=1)
        print(scn, plots(scn, d, floor))
        for r in rows:
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != "vs_ref"})
            for s in (50, 100, 500, 1000, 2000, 3000, 4000):
                x = r.get("vs_ref", {}).get(s)
                if x:
                    print("   ", s, "dx_mean %.3g dx_med %.3g corr %.4f rq %s" % (x["dx_mean"], x["dx_median"], x["density_corr"], [round(q, 4) for q in x["rq_ratio"]]))
        if floor:
            for x in floor["vs_ref"]:
                if x["step"] in (50, 100, 500):
                    print("floor", x["step"], "dx_mean %.3g corr %.4f rq %s" % (x["dx_mean"], x["density_corr"], [round(q, 4) for q in x["rq_ratio"]]))


if __name__ == "__main__":
    main()
