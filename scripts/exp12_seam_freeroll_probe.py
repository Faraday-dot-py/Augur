"""Force jump across the x=0 seam for the massless probe scan in results/seam_freeroll/seam_all.json: owner-window force step across the seam vs the step between the
same two positions from one window (the one on each side), vs the exact force step. Probe at 1-unit spacing x=-6..6, owner flips at x=0."""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/seam_freeroll/seam_all.json"))
for arm, seeds in r.items():
    for sd, v in seeds.items():
        for key, rows in v.get("probe", {}).items():
            ownF = np.array([row["a"][str(row["owner"])] for row in rows])
            ex = np.array([row["exact"] for row in rows])
            i = [k for k, row in enumerate(rows) if row["x"] == 0.0][0]
            lo, hi = str(rows[i - 1]["owner"]), str(rows[i]["owner"])
            sameA = np.array(rows[i]["a"][lo]) - np.array(rows[i - 1]["a"][lo]) if lo in rows[i]["a"] else None
            jump = ownF[i] - ownF[i - 1]
            nb = np.mean([np.linalg.norm(ownF[j] - ownF[j - 1]) for j in (i - 2, i + 1)])
            mag = np.linalg.norm(ex[i])
            print(f"{arm} {sd} {key}: |a_exact| {mag:.3f}  owner step across seam {np.linalg.norm(jump):.3f}  neighbour steps {nb:.3f}  same-window(left) step {np.linalg.norm(sameA) if sameA is not None else float('nan'):.3f}  exact step {np.linalg.norm(ex[i] - ex[i - 1]):.3f}  window-to-window gap at x=0 {np.linalg.norm(np.array(rows[i]['a'][lo]) - np.array(rows[i]['a'][hi])) if lo in rows[i]['a'] else float('nan'):.3f}")
