"""Summary for job 4850 (curiosity): activation patching with native reference. Per pair and k: top shift fraction, random-draw mean and max, absolute change rms (top, random mean), native source-target gap."""
import json

import numpy as np

R = "results/"
for name in ("sae_activation_patching_native_reference_dec0_s4738_mesh_512", "sae_activation_patching_native_reference_dec1_s4738_mesh_64"):
    d = json.load(open(R + name + ".json"))
    print(name, "n_live", d["n_live"], "ks", d["ks"], "scenes", d["scenes"])
    for pair, r in d["pairs"].items():
        sh, dl = r["shift_fraction_median"], r["delta_rms_median"]
        parts = [f"{pair}{' [CONTROL]' if r['control_same_law_shuffled_velocity'] else ''} gap {r['source_target_native_gap_rms_median']:.3f} recon-only shift {sh['sae_recon_no_patch']:+.3f} dRMS {dl['sae_recon_no_patch']:.3f}"]
        for k in d["ks"]:
            rnd = [sh[f"rand{k}_{i}"] for i in range(d["draws"])]
            rd = [dl[f"rand{k}_{i}"] for i in range(d["draws"])]
            parts.append(f"k{k} top {sh[f'top{k}']:+.3f} (dRMS {dl[f'top{k}']:.3f}) rand {np.mean(rnd):+.3f} max {max(rnd):+.3f} (dRMS {np.mean(rd):.3f})")
        print(" ", " | ".join(parts))
