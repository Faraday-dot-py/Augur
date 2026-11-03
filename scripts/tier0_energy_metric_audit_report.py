"""Prints the headline tables of results/tier0_tiled_closed_loop/energy_metric_audit.json."""
import json
import sys

r = json.load(open(sys.argv[1]))
arms = ["truth_self", "oracle_force_dt0.1_verlet", "baseline_tiled", "matched60k_noexchange", "matched60k_halo", "new_halo"]
for sd, v in r.items():
    b = v["baseline_tiled"]["100"]["dE_vs_E0_mean"]
    print(f"seed {sd}  mean E0 {v['truth_E0']:.1f}")
    print(f"{'arm':28s} dE100/E0  xBL  signed  dE|Et(t)  sim_signed  dKE   dPE   same   cross  r<4   r>=4  outside speedmax")
    for a in arms:
        x = v[a]["100"]
        print(f"{a:28s} {x['dE_vs_E0_mean']:.4f} {x['dE_vs_E0_mean'] / b:5.2f} {x['signed_vs_E0_mean']:+.4f} {x['dE_vs_Etruth_t_mean']:.4f} {x['signed_vs_Etruth_t_mean']:+.4f} "
              f"{x['dKE_vs_truth_t']:+.4f} {x['dPE_vs_truth_t']:+.4f} {x['dPE_same_tile_vs_truth_t']:+.4f} {x['dPE_cross_tile_vs_truth_t']:+.4f} {x['dPE_r_lt4_vs_truth_t']:+.4f} {x['dPE_r_ge4_vs_truth_t']:+.4f} {x['outside_mean']:.2f} {x['speed_max_mean']:.2f}")
    for a in arms[2:]:
        print(a, "signed-vs-Etruth(t) profile every 5 ticks:", " ".join(f"{q:+.3f}" for q in v[a]["profile_signed_vs_Etruth_t"][::2]))
    print()
