"""Markdown table for option C: python scripts/scatter_contact_c_table.py [dir]  (reads eval_{E_zs13,B2_13,A2_13,C}.json)."""
import json
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results/contact"
TAGS = [("E_zs13", "E zero-shot"), ("B2_13", "B2 (ref)"), ("A2_13", "A2 (ref)"), ("C", "C"), ("C40", "C (k_end 40)")]
FAMS = ["c_id", "c_big", "c_small", "c_pos", "c_count", "c_nonconvex", "c_cont", "box15", "box_var", "post_circle", "post_rect", "disc", "post_poly"]
D = {t: json.load(open(f"{R}/eval_{t}.json")) for t, _ in TAGS}


def m(fam, key, t):
    return float(np.mean([s[key][t] for s in fam["seeds"].values()]))


print("| family | model | err@5 | err@20 | err@100 | tunnel@100 % | KE ratio@100 | dE@100 model/truth | mom err@20 |")
print("|---|---|---|---|---|---|---|---|---|")
for f in FAMS:
    for t, name in TAGS:
        x = D[t]["families"][f]
        print(f"| {f} | {name} | {m(x, 'err', '5'):.2f} | {m(x, 'err', '20'):.2f} | {m(x, 'err', '100'):.4g} | {100 * m(x, 'tunnel_model', '100'):.1f} | {m(x, 'ke_ratio', '100'):.3g} | {m(x, 'dE_model', '100'):.3g}/{m(x, 'dE_truth', '100'):.3g} | {m(x, 'mom_err', '20'):.3g} |")
    x = D["C"]["families"][f]
    print(f"| {f} | floor/ballistic/truth-tunnel | {m(x, 'floor_pert1e-5', '5'):.2g} | {m(x, 'floor_pert1e-5', '20'):.2f} (ball. {m(x, 'ballistic_gravity', '20'):.1f}) | {m(x, 'floor_pert1e-5', '100'):.2f} | {100 * m(x, 'tunnel_truth', '100'):.1f} | | | |")
f = lambda x: "-" if x is None else f"{x:.2f}"
for k in ("restitution_wall", "restitution_circle"):
    print(f"\n{k} (v: model e_eff; truth ~1)")
    for t, name in TAGS:
        print(f"- {name}:", {v: f((x or {}).get("model")) for v, x in D[t][k].items()})
