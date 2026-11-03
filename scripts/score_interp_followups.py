"""Score the row 1 / row 2 follow-ups against the lines pre-registered in docs/theory/supervisor-log.md (2026-11-03, 'PRE-REGISTRATION of follow-ups').
Usage: python scripts/score_interp_followups.py results/law_recovery_followup.json results/pm_baseline_more_scenes_and_families.json results/pm_baseline_density_scaled.json"""
import json
import subprocess
import sys

lr = json.load(open(sys.argv[1]))
print("ROW 1 per-model p_err / eps_err (placement seeds 4738,4739,4740 | dense | bootstrap mean [ci95] | ranges)")
for name, laws in lr["models"].items():
    for law, r in laws.items():
        ps = [r["placement_seeds"][s] for s in ("4738", "4739", "4740")]
        b = r.get("bootstrap")
        print(f"{name:12s} {law:7s} p_err {[round(x['p_err'], 3) for x in ps]} eps_err {[round(x['eps_err'], 3) for x in ps]} dense {r['dense_grid_56']['p_err']:.3f}/{r['dense_grid_56']['eps_err']:.3f}"
              + (f" boot p {b['p_err_mean']:.3f} [{b['p_err_ci95'][0]:.3f},{b['p_err_ci95'][1]:.3f}] eps {b['eps_err_mean']:.3f}" if b else "")
              + " ranges p_err " + str({k: round(v['p_err'], 3) for k, v in r['ranges'].items() if v}))
m = lr["models"]["mixed_s4738"]
est = {}
for law in ("pow1", "pow2", "yuk8", "yuk128"):
    r = m[law]
    e = [r["placement_seeds"][s] for s in ("4738", "4739", "4740")] + [r["dense_grid_56"]]
    b = r.get("bootstrap")
    est[law] = [(x["p_err"], x["eps_err"]) for x in e] + ([(b["p_err_mean"], b["eps_err_mean"])] if b else [])
pow1_fail = sum(p > 0.10 for p, _ in est["pow1"])
others_ok = all(all(p <= 0.10 and e <= 0.20 for p, e in est[l][:4]) for l in ("pow2", "yuk8", "yuk128"))
all_ok = all(all(p <= 0.10 and e <= 0.20 for p, e in v[:4]) for v in est.values())
kill = any(sum(p > 0.30 or e > 0.30 for p, e in v) > len(v) / 2 for v in est.values())
print("pow1 re-estimates over 0.10:", pow1_fail, "of", len(est["pow1"]), "| others ok on 4 estimates:", others_ok)
print("ROW1 CLASSIFICATION", "KILL" if kill else "PASS_CONFIRMED" if all_ok else "CONFIRMED_PARTIAL_3_OF_4" if pow1_fail >= 3 and others_ok else "UNRESOLVED")
print("YUKAWA range fits (mixed_s4738)")
for law in ("yuk8", "yuk32", "yuk128"):
    r = m[law]
    print(law, "free", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["free_fit"].items()}, "| fixed", {k: round(v, 3) for k, v in r["lam_fixed_to_truth"].items() if v is not None and k in ("p_err", "eps_err", "rms_log_resid")},
          "| exact-truth free fit lam", r["exact_truth_curve_free_fit"]["lam"], "resid", round(r["exact_truth_curve_free_fit"]["rms_log_resid"], 4))
ms = json.load(open(sys.argv[2]))
print("ROW 2 N=1000, 64 scenes: PM/learned (median, p90)")
for f in ("plummer", "two_clump", "exp_disk", "hernquist", "three_clump"):
    row = ms[f"{f}_n1000"]
    L, P = row["learned_warm"]["norm_by_rms"], row["plain_pm"]["norm_by_rms"]
    print(f"{f:12s} learned {L['median']:.4f}/{L['p90']:.4f} PM {P['median']:.4f}/{P['p90']:.4f} ratio {P['median'] / L['median']:.2f}/{P['p90'] / L['p90']:.2f}")
print(subprocess.run([sys.executable, "scripts/score_pm_baseline.py", sys.argv[2]], capture_output=True, text=True).stdout)
ds = json.load(open(sys.argv[3]))
print("density-scaled series")
for k, row in ds.items():
    if isinstance(row, dict) and "plain_pm" in row:
        L, P = row["learned_warm"]["norm_by_rms"], row["plain_pm"]["norm_by_rms"]
        print(f"{k:16s} ratio {P['median'] / L['median']:.2f}/{P['p90'] / L['p90']:.2f}  learned {L['median']:.4f}/{L['p90']:.4f} PM {P['median']:.4f}/{P['p90']:.4f}")
