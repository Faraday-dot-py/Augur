import json, sys, collections

d = sys.argv[1]
c = json.load(open(f"{d}/components.json"))
s = json.load(open(f"{d}/sweep.json"))
g = json.load(open(f"{d}/gap.json"))
out = {}

frac = {}
for k, v in s.items():
    ck, n, fam, sd, r = k.split("|")
    e = frac.setdefault(f"{ck}|{n}", [0, 0])
    e[1] += 1
    e[0] += v["bad_tick"] is not None or v["exception"] is not None
out["h4_diverged_over_total"] = frac

variants = ["grid-zeroed", "far-zeroed", "pair-zeroed", "grid-capped-10x-exact-max", "far-capped-10x-exact-max", "pair-capped-10x-exact-max", "all-capped-10x-exact-max", "cell-cap-256"]
h1 = {}
for ck in ("25k", "30k", "both"):
    ks = [k for k, v in c.items() if "ablations" in v and v["ablations"]["baseline"]["bad_tick"] is not None and (ck == "both" or k.startswith(ck + "|"))]
    h1[ck] = {"diverged": len(ks), **{vn: sum(c[k]["ablations"][vn]["bad_tick"] is None for k in ks) / len(ks) for vn in variants}}
out["h1_cure_fraction"] = h1

ks = [k for k, v in c.items() if "ablations" in v and v["ablations"]["baseline"]["bad_tick"] is not None]
healthy_same = [k for k, v in c.items() if k.split("|")[1] == "1000" and k.split("|")[0] in ("25k", "30k") and v["ablations"]["baseline"]["bad_tick"] is None]
healthy_early = [k for k, v in c.items() if k.split("|")[1] == "1000" and k.split("|")[0] in ("10k", "20k")]
r = lambda kk, ch: [c[k]["ratio_to_exact_max"][ch] for k in kk]
out["h1_tick0_grid_ratio"] = {
    "diverged_min": min(r(ks, "grid")), "diverged_n_below_10": sum(x < 10 for x in r(ks, "grid")),
    "healthy_same_ckpt_n": len(healthy_same), "healthy_same_ckpt_n_ge5": sum(x >= 5 for x in r(healthy_same, "grid")), "healthy_same_ckpt_max": max(r(healthy_same, "grid")),
    "healthy_10k_20k_max": max(r(healthy_early, "grid")), "healthy_10k_20k_n_ge5": sum(x >= 5 for x in r(healthy_early, "grid"))}
out["h2_cells_over_cap_max_in_diverged"] = max(c[k]["cells_over_cap"] for k in ks)
tot = r(ks, "total")
out["h3"] = {"diverged": len(ks), "tick0_total_ge10": sum(x >= 10 for x in tot), "tick0_total_lt3": sum(x < 3 for x in tot),
             "tf_vs_rollout_tick1_within10pct": sum(abs(c[k]["teacher_forced_ticks_1_to_5"][0]["pos_err"] - c[k]["rollout_ticks_1_to_5"][0]["pos_err"]) <= 0.1 * c[k]["rollout_ticks_1_to_5"][0]["pos_err"] for k in ks),
             "tf_vs_rollout_tick2_median_ratio": sorted(c[k]["rollout_ticks_1_to_5"][1]["pos_err"] / c[k]["teacher_forced_ticks_1_to_5"][1]["pos_err"] for k in ks)[len(ks) // 2] if all(len(c[k]["rollout_ticks_1_to_5"]) > 1 for k in ks) else None}
out["h6_gap"] = {k: (v["train_err20"], v["val_err20"], v["val_over_train"]) for k, v in g.items()}
print(json.dumps(out, indent=1))
