"""Print the numbers for docs/debugging/adaptive-eval-interp.md from results/interp_*.json."""
import json
import os

import numpy as np

NAMES = ["analytic", "inv_distance", "yukawa30", "plw0.5", "plw1.5", "plw3"]
THEORY_P = {"analytic": 2, "inv_distance": 1, "yukawa30": 2, "plw0.5": 0.5, "plw1.5": 1.5, "plw3": 3}


def s1():
    if not os.path.exists("results/interp_stage1.json"):
        return
    r = json.load(open("results/interp_stage1.json"))
    print("## rho exponent (fit rho<=0.4), per dist quantile (0.2/0.5/0.8)")
    for n, v in r["rho_sweep"].items():
        print(n, "dists", [round(d, 1) for d in v["dists"]], "truth", np.round(v["truth_slope"], 2), "taylor", np.round(v["taylor_slope"], 2), "est", np.round(v["est_slope"], 2))
    print("## partial dependence slopes d pred / d log rho (size_only, quad_only, both)")
    for n, v in r["partial_dep"].items():
        print(n, {k: [round(x, 2) for x in d.values()] for k, d in v.items()})
    print("## anisotropy (mean log label): truth / taylor / est")
    for n, v in r["aniso"].items():
        print(n, "dist", round(v["dist"], 1))
        for k, d in v.items():
            if isinstance(d, dict):
                print("  ", k, round(d["truth"], 2), round(d["taylor"], 2), round(d["est"], 2))
        for sc in ("tinyA", "both"):
            t1, t4 = v[f"{sc}_k1_along"]["truth"], v[f"{sc}_k4_along"]["truth"]
            e1, e4 = v[f"{sc}_k1_along"]["est"], v[f"{sc}_k4_along"]["est"]
            print("   ", sc, "ratio elongated(k4 along)/round truth", round(float(np.exp(t4 - t1)), 2), "est", round(float(np.exp(e4 - e1)), 2))
    print("## label stats OLS [1, log_rho, log_dist, log_count, meanlogq, contrast, corr], r2")
    for n, v in r["label_stats"].items():
        print(n, np.round(v["ols"], 2), round(v["r2"], 3))
    print("## symmetry")
    for n, v in r["symmetry"].items():
        print(n, "exch", v["exchange_max"], "rot mean/max", v["rotation_mean"], v["rotation_max"], "truth rot", v["rotation_truth_mean"], "rmse", round(v["pred_vs_truth_rmse"], 3))
        for f, d in v["scale"].items():
            print("   scale", f, {k: round(x, 3) for k, x in d.items()})
    print("## dist sweep")
    for n, v in r["dist_sweep"].items():
        print(n, "dist", np.round(v["dists"], 1))
        print("  truth", np.round(v["truth"], 2))
        print("  taylor", np.round(v["taylor"], 2))
        for k, e in v["est"].items():
            print("  est", k, np.round(e, 2))


def dec():
    if not os.path.exists("results/interp_decode.json"):
        return
    r = json.load(open("results/interp_decode.json"))
    print("## decode: median p relerr (max over 3 seeds) by method|N|sigma")
    for kt, rec in r["decode"].items():
        print(kt)
        for meth in ("pair_p", "pair_full", "vec_p", "vec_full", "scal_p", "scal_full"):
            row = []
            for sg in (0.0, 0.01):
                for N in (4, 16, 64, 256):
                    d = rec.get(f"{meth}|N{N}|s{sg}")
                    row.append("%.3g/%.3g" % (d["p_relerr_median"], d["p_relerr_max"]) if d else "-")
            print("  ", meth, " ".join(row))
        d = rec.get("vec_full|N256|s0.01")
        if d:
            print("   vec_full N256 s.01 eps_med, kappa_med", round(d["eps_med"], 3), round(d["kappa_med"], 4))
    print("## est route")
    for kt, rec in r["est_route"].items():
        print(kt, "est rmse", round(rec["est_vs_truth_rmse"], 3), "bias", round(rec["est_bias"], 3))
        for k, d in rec.items():
            if isinstance(d, dict):
                print("  ", k, d)


def maps():
    if not os.path.exists("results/interp_maps.json"):
        return
    r = json.load(open("results/interp_maps.json"))
    for kn, e in r["runs"].items():
        print(kn, json.dumps({k: e[k] for k in ("est", "geo035", "geo_matched")}))
    for k, v in r["matrix"].items():
        print(k, v)
    for k, v in r["rotation"].items():
        print("rot", k, {m: {a: round(b, 4) for a, b in d.items() if a in ("cost", "rel_l2", "p99")} for m, d in v.items()})


if __name__ == "__main__":
    s1()
    dec()
    maps()
