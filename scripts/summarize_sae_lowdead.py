import json
import sys

r = json.load(open(sys.argv[1]))
for part in [p for p in ("mesh", "pair") if p in r]:
    for k, v in r[part].items():
        if k.startswith("dict_"):
            print(part, k, "nmse_heldout", v["recon_nmse_heldout"], "l0", round(v["l0_heldout"], 2), "heldout_dead", round(v["heldout_dead_fraction"], 3), "live", v["n_live_heldout"],
                  "law_eta2 max", v["law_eta2_live_max"], "n>0.5", v["n_live_law_eta2_gt_0.5"],
                  {kk: vv for kk, vv in v.items() if kk.endswith("abscorr_live_max")})
