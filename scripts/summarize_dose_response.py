import json
import sys

import numpy as np

for p in sys.argv[1:]:
    r = json.load(open(p))
    print(p, r["key"], "live", r["n_live"], "ks", r["ks"])
    for law, arms in r["errors"].items():
        row = [f"{law:7s} rec {arms['sae_reconstruction']['median']:.3f}"]
        for k in r["ks"]:
            rnd = np.array([arms[f"rand{k}_{d}"]["median"] for d in range(r["draws"])])
            row.append(f"k{k} top {arms[f'top{k}']['median']:.3f} rand {rnd.mean():.3f}[{rnd.min():.3f},{rnd.max():.3f}]")
        print(" | ".join(row))
