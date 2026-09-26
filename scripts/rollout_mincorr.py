import json

for scn, pre in (("flyby", "rollout_flyby_"), ("flyby12000", "rollout_s12000_"), ("uniform", "rollout_uni_")):
    print(scn)
    for p in ("adaptive", "geo_audit", "est", "bh", "mesh1024", "mesh2048"):
        try:
            v = json.load(open(f"results/{pre}{p}.json"))["vs_ref"]
        except FileNotFoundError:
            continue
        c = min(v, key=lambda x: x["density_corr"])
        rmax = max(abs(q - 1) for x in v for q in x["rq_ratio"][2:])
        print("  %-10s min corr %.3f at step %d  max |rq75..99 ratio-1| %.3f  mean corr %.3f" % (p, c["density_corr"], c["step"], rmax, sum(x["density_corr"] for x in v) / len(v)))
