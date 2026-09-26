import json

for scn in ("flyby", "flyby12000", "uniform"):
    print(scn)
    for r in json.load(open(f"results/rollout_summary_{scn}.json")):
        print("  %-10s dE_final %.4f dEvsEx %+.4f Pmax %.1e Ldrift %.1e comdrift %.1e step_s %.4f cost %s events %s" % (
            r["provider"], r["dE_over_E0_final"], r["dE_vs_exact_final"], r["P_rel_max"], r["L_drift_rel_max"], r["com_drift_max"], r["step_s"], r.get("cost"), len(r.get("events", []))))
