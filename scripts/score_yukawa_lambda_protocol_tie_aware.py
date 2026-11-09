import json
import sys

TOL = 0.02
r = json.load(open(sys.argv[1]))
m = r["models"][sys.argv[2] if len(sys.argv) > 2 else "mixed_s4738"]
wrong = 0
bad_err = 0
cells = 0
for law, seeds in m.items():
    for s, c in seeds.items():
        res = c["resid_by_candidate"]
        lo = min(res.values())
        tied = sorted(k for k, v in res.items() if v <= lo * (1 + TOL))
        ok = c["true"] in tied and len(tied) <= 2
        cells += 1
        wrong += not ok
        bad_err += (c["p_err"] > 0.10) or (c["eps_err"] > 0.20)
        print(law, s, "true", c["true"], "tied set", tied, "correct", ok, "selected", c["selected"], "p_err %.3f eps_err %.3f" % (c["p_err"], c["eps_err"]))
verdict = "PASS" if wrong == 0 and bad_err == 0 else "KILL" if wrong >= 5 else "INCONCLUSIVE_PARTIAL"
print("cells", cells, "wrong", wrong, "cells failing p/eps lines", bad_err, "tolerance", TOL, "VERDICT", verdict)
