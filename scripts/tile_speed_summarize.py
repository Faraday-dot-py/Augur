import json

d = json.load(open("results/tile_speed_test.json"))
for key, r in d.items():
    nz = r["noise_1e-6"]["20"]
    print(f"{key} balls={r['balls']} vmax_t0={r['max_speed_t0']:.1f} noise20 mean={nz['mean']:.2e} max={nz['max']:.2e}")
    for name, t in r["tiled"].items():
        dv = max(t["one_step"][k]["dv"]["max"] for k in t["one_step"])
        fr = max(t["one_step"][k]["dv"]["frac_gt_1e-3"] for k in t["one_step"])
        sp = max(t["one_step"][k]["max_speed"] for k in t["one_step"])
        ro = t["rollout"]["20"]
        print(f"  {name:9s} 1step dv max={dv:.1e} frac>1e-3={fr:.4f} (vmax {sp:.0f}) | roll20 mean={ro['mean']:.1e} max={ro['max']:.1e} frac={ro['frac_gt_1e-3']:.4f}")
