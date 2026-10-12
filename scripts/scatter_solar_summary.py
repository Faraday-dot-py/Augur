"""Print the solar-system run table from results/solar/solar.json (and the selection table from solar_select.json)."""
import json
import sys

sel = json.load(open("results/solar/solar_select.json"))
print("SELECTION (planetary benchmark, mean of seeds 9100/9200/9300, 1000 ticks)")
for k, v in sel.items():
    m = v["mean"]
    print(f"{k:12s} period err {m['period_err_median']:.3f} lost {m['planets_lost_frac']:.3f} err100/r0 {m['err100_over_r0_median']:.3f} dE/E {m['dE_over_E_1000']:.3g} dL/L {m['dL_over_L_1000']:.3g}")
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/solar/solar.json"))
print("\nSOLAR IC", r["n_bodies"], "bodies", r["kind_counts"])
for name, c in r["cands"].items():
    print("==", name)
    for tag in ("planets", "belt"):
        for t, m in c[tag].items():
            print(f"  {tag:7s} t={t:5s} period err {m['period_err_median']:.3f} lost {m['lost_frac']:.3f} pos err med {m['pos_err_median']:.2f} r/r_truth {m['radius_ratio_median']:.2f}")
    for t, m in c["conservation"].items():
        print(f"  t={t:5s} dE model {m['dE_model']:+.3g} truth {m['dE_truth']:+.2g}  dL model {m['dL_model']:+.3g} truth {m['dL_truth']:+.2g}")
    print("  moon", c["moon"], "star drift model/truth", round(c["star_drift_model"], 3), round(c["star_drift_truth"], 4))
    print("  per planet (a0, m, period err, r/r0 model, r/r0 truth, lost):", [(round(p["a0"], 1), p["mass"], round(p["period_err_T"], 2), round(p["r_over_r0_T_model"], 2), round(p["r_over_r0_T_truth"], 2), p["lost"]) for p in c["per_planet"]])
