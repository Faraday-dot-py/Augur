"""Score results/law_recovery_probe.json against the pre-registration (docs/theory/supervisor-log.md, 2026-11-03, audit row 1).
PASS: primary model mixed_s4738, full-range full-model fit, laws pow2 (gravity), pow1 (1/r), yuk8 and yuk128 (Yukawa): p within 10% and eps within 20% of truth.
KILL: any of those laws has p or eps error > 30%.  PARTIAL: not PASS/KILL-free but the pair-term-only near-range fit (d <= 2) passes on all four laws.
Otherwise INCONCLUSIVE_PARTIAL. Usage: python scripts/score_law_recovery.py results/law_recovery_probe.json"""
import json
import sys

LAWS = ("pow2", "pow1", "yuk8", "yuk128")


def main():
    r = json.load(open(sys.argv[1]))
    prim = r["models"]["mixed_s4738"]
    full_ok, kill, near_ok = [], [], []
    for law in LAWS:
        f = prim[law]["fit_full"]["full"]
        pe, ee = f["p_err"], f["eps_err"]
        full_ok.append(pe <= 0.10 and ee <= 0.20)
        kill.append(pe > 0.30 or ee > 0.30)
        g = prim[law]["fit_pair"]["near"]
        near_ok.append(g is not None and g["p_err"] <= 0.10 and g["eps_err"] <= 0.20)
        print(law, "full p_err %.3f eps_err %.3f | pair-only near p_err %s eps_err %s" % (pe, ee, g and round(g["p_err"], 3), g and round(g["eps_err"], 3)))
    for name, m in r["models"].items():
        for law, rec in m.items():
            if law == "ckpt" or "fit_full" not in rec:
                continue
            f = rec["fit_full"]["full"]
            print("  %-14s %-7s p %.3f (err %.3f) eps %.3f (err %.3f) lam %s | truth-fit p %.3f eps %.3f" % (
                name, law, f["p"], f["p_err"], f["eps"], f["eps_err"], f["lam"], rec["fit_truth"]["full"]["p"], rec["fit_truth"]["full"]["eps"]))
    if all(full_ok):
        v = "PASS"
    elif any(kill):
        v = "KILL"
    elif all(near_ok):
        v = "PARTIAL_NEAR_FIELD_ONLY"
    else:
        v = "INCONCLUSIVE_PARTIAL"
    print("VERDICT", v)


if __name__ == "__main__":
    main()
