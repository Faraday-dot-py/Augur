import json
import sys

import numpy as np

for tag, base in zip(sys.argv[1::2], sys.argv[2::2]):
    r = json.load(open(base + ".json"))["mesh"]
    z = np.load(base + ".npz")
    for k, v in r.items():
        if not k.startswith("dict_"):
            continue
        e = z[f"mesh_{k[5:]}_law_eta2"]
        live = e > 0
        print(tag, k, "heldout_dead", round(v["heldout_dead_fraction"], 3), "nmse_heldout", "%.2e" % v["recon_nmse_heldout"], "live", int(live.sum()),
              "eta2 median(live) %.2f" % float(np.median(e[live])), "frac live >0.5 %.2f" % float((e[live] > 0.5).mean()), "eta2 max %.2f" % float(e.max()),
              "|corr| cell mass max %.2f" % float(z[f"mesh_{k[5:]}_log_cell_mass_abscorr"].max()))
