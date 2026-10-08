import json

import numpy as np

R = "results/contact"
TAGS = [("E_zs", "E zero-shot"), ("augur1", "bounce_ft1"), ("B2", "B (SDF+MLP)"), ("A2", "A (kernel)")]
FAMS = ["box15", "box_var", "post_circle", "post_rect", "disc", "post_poly"]


def mean(fam, key, t):
    return float(np.mean([s[key][t] for s in fam["seeds"].values()]))


print("| family | model | err@5 | err@20 | err@100 | tunnel@100 % | KE ratio@100 |")
print("|---|---|---|---|---|---|---|")
for f in FAMS:
    for tag, name in TAGS:
        fam = json.load(open(f"{R}/eval_{tag}.json"))["families"][f]
        ke = np.mean([s["ke_ratio"]["100"] for s in fam["seeds"].values()]) if "ke_ratio" in next(iter(fam["seeds"].values())) else float("nan")
        print(f"| {f} | {name} | {mean(fam, 'err', '5'):.2f} | {mean(fam, 'err', '20'):.2f} | {mean(fam, 'err', '100'):.4g} | {100 * mean(fam, 'tunnel_model', '100'):.1f} | {ke:.3g} |")
    fam = json.load(open(f"{R}/eval_B2.json"))["families"][f]
    print(f"| {f} | floor/ballistic/truth-tunnel | {mean(fam, 'floor_pert1e-5', '5'):.2g} | {mean(fam, 'floor_pert1e-5', '20'):.2f} (ball. {mean(fam, 'ballistic_gravity', '20'):.1f}) | {mean(fam, 'floor_pert1e-5', '100'):.2f} | {100 * mean(fam, 'tunnel_truth', '100'):.1f} | |")
