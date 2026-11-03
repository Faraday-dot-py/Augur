import json
import sys

r = json.load(open(sys.argv[1]))
m = r["models"]["mixed_s4738"]
wrong = 0
bad_err = 0
for law, seeds in m.items():
    for s, c in seeds.items():
        wrong += not c["correct"]
        bad_err += (c["p_err"] > 0.10) or (c["eps_err"] > 0.20)
        print(law, s, "selected", c["selected"], "true", c["true"], "p_err %.3f eps_err %.3f" % (c["p_err"], c["eps_err"]), {k: round(v, 3) for k, v in c["resid_by_candidate"].items()},
              "free lam", c["free_fit"]["lam"])
verdict = "PASS" if wrong == 0 and bad_err == 0 else "KILL" if wrong >= 5 else "INCONCLUSIVE_PARTIAL"
print("wrong selections", wrong, "of 9; cells failing p/eps lines", bad_err, "VERDICT", verdict)
for law, seeds in r["models"].get("mixed_s4739", {}).items():
    print("context 4739", law, [(c["selected"], round(c["p_err"], 3)) for c in seeds.values()])
