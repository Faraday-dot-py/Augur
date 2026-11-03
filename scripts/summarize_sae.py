"""Print the headline rows of an sae_interp JSON. Usage: python scripts/summarize_sae.py results/sae_interp_s4738_slowcontrol.json"""
import json
import sys

r = json.load(open(sys.argv[1]))
for part in ("mesh", "pair"):
    for k, v in r[part].items():
        if not k.startswith("dict_"):
            continue
        print(part, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "top_law_features"})
        t = v["top_law_features"][0]
        print("   top law feature", t["feature"], round(t["eta2"], 3), {a: round(b, 2) for a, b in t["mean_act_per_law"].items()})
