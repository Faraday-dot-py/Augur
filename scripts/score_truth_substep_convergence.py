import json
import sys

import numpy as np

for path in sys.argv[1:]:
    rows = json.load(open(path))
    print(path)
    for seed in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == seed]
        pos = {r["substeps"]: r["pos_vs_ias15_max"] for r in rs}
        de = {r["substeps"]: r["dE_vs_ias15_max"] for r in rs}
        subs = sorted(pos)
        ratios = [pos[a] / pos[b] for a, b in zip(subs[:-1], subs[1:])]
        order = [float(np.log2(x)) / 1.0 for x in ratios]
        print(f"seed {seed}: pos " + " ".join(f"{s}:{pos[s]:.2e}" for s in subs))
        print("   dE " + " ".join(f"{s}:{de[s]:.2e}" for s in subs))
        print("   pos ratio per doubling " + " ".join(f"{x:.2f}" for x in ratios))
        r4 = [r for r in rs if r["substeps"] == 4][0]
        print("   t4 lagrangian rel diff final r10/r50/r90 " + " ".join(f"{x:.1e}" for x in r4["lagrangian_rel_diff_final"]))
