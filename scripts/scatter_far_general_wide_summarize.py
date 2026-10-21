"""Mean over held-out seeds of tiled+far err@5/10/20 per kernel spec from far_general_wide.json."""
import json
import sys
from collections import defaultdict

rows = json.load(open(sys.argv[1]))
acc = defaultdict(list)
for r in rows:
    if r["variant"] == "far":
        acc[r["spec"]].append(r["err"])
for spec, errs in acc.items():
    m = {k: sum(e[k] for e in errs) / len(errs) for k in errs[0]}
    print(spec, " ".join(f"err@{k} {v:.3f}" for k, v in m.items()), f"n={len(errs)}")
