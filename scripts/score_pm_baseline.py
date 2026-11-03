"""Score results/pm_baseline_force_error.json against the pre-registration (docs/theory/supervisor-log.md, 2026-11-03, audit row 2).
Headline N=1000, learned_warm vs plain_pm, normalized error. PASS: PM/learned >= 2 at median AND p90 on all three families.
KILL: PM/learned p90 < 1.2 on >= 2 of 3 families. Else INCONCLUSIVE_PARTIAL.
Usage: python scripts/score_pm_baseline.py results/pm_baseline_force_error.json [family,...] [N]"""
import json
import sys

r = json.load(open(sys.argv[1]))
fams = sys.argv[2].split(",") if len(sys.argv) > 2 else ["plummer", "two_clump", "exp_disk"]
n = sys.argv[3] if len(sys.argv) > 3 else "1000"
ok, kill = [], []
for f in fams:
    row = r[f"{f}_n{n}"]
    L, P = row["learned_warm"]["norm_by_rms"], row["plain_pm"]["norm_by_rms"]
    rm, r9 = P["median"] / L["median"], P["p90"] / L["p90"]
    ok.append(rm >= 2 and r9 >= 2)
    kill.append(r9 < 1.2)
    print(f"{f:12s} N={n} learned med {L['median']:.4f} p90 {L['p90']:.4f} | PM med {P['median']:.4f} p90 {P['p90']:.4f} | ratio med {rm:.2f} p90 {r9:.2f}")
print("VERDICT", "PASS" if all(ok) else "KILL" if sum(kill) >= 2 else "INCONCLUSIVE_PARTIAL")
