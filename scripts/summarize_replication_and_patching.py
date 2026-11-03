"""Summaries for job 4843 (curiosity): dose-response top-k vs random-draw ranges per law, and activation patching shift fractions."""
import json
import sys

import numpy as np

R = "results/"
for name in ("sae_ablation_dose_response_dec2_s4739checkpoint_mesh_128", "sae_ablation_dose_response_dec2_s4739checkpoint_mesh_256", "sae_ablation_dose_response_dec0_s4738_heldout16_mesh_256"):
    d = json.load(open(R + name + ".json"))
    print(name, "n_live", d["n_live"], "ks", d["ks"], "scenes", d.get("n_scenes_force"), "heldout_rank", d.get("heldout_rank"))
    for k in d["ks"]:
        rows = []
        for law, e in d["errors"].items():
            nat = e["native"]["median"]
            top = e[f"top{k}"]["median"]
            rnd = [e[f"rand{k}_{i}"]["median"] for i in range(d["draws"])]
            distinct = len({tuple(sorted(v)) for n, v in d["draw_members"].items() if n.startswith(f"rand{k}_")})
            rows.append((law, nat, top, float(np.mean(rnd)), float(np.max(rnd)), top > max(rnd)))
        print(f" k={k} distinct_draws={distinct} top>all_random in {sum(r[5] for r in rows)}/{len(rows)} laws; top/native ratio median {np.median([r[2]/r[1] for r in rows]):.2f}; rand_mean/native median {np.median([r[3]/r[1] for r in rows]):.2f}")
    print(" top features k=5:", [(f["feature"], round(f["eta2"], 3), round(f["freq"], 3)) for f in d["top_features"]["5"]])
for name in ("sae_activation_patching_dec2_s4739checkpoint_mesh_256", "sae_activation_patching_dec0_s4738_mesh_256"):
    d = json.load(open(R + name + ".json"))
    print(name, "n_live", d["n_live"], "scenes", d["scenes"])
    for pair, r in d["pairs"].items():
        sh = r["shift_fraction_median"]
        out = []
        for k in d["ks"]:
            rnd = [sh[f"rand{k}_{i}"] for i in range(d["draws"])]
            out.append(f"k{k} top {sh[f'top{k}']:+.3f} rand mean {np.mean(rnd):+.3f} [{min(rnd):+.3f},{max(rnd):+.3f}]")
        print(" ", pair, " | ".join(out), "| tgt_native err to tgt/src", round(r["error_median"]["target_native"]["to_target"], 3), round(r["error_median"]["target_native"]["to_source"], 3), "src_native err to tgt", round(r["error_median"]["source_native"]["to_target"], 3))
