import glob
import json
import os
import sys

res = sys.argv[1]
SEED = "9000"


def load(exp):
    return {os.path.basename(p)[len(exp) + 1:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{res}/{exp}_*.json")) if not p.endswith("_train.json")}


print("## per-channel RMS @5/10/20 (pos_x pos_y vel_x vel_y), seed 9000, exp A and B; mass drift@100 (ml), curl/total, self-force")
for exp in "AB":
    for n, r in load(exp).items():
        if SEED not in r or "chan_rms_at_5_10_20" not in r[SEED]:
            continue
        c = r[SEED]["chan_rms_at_5_10_20"]
        print(exp, n, " ".join(f"{k}:{c[k][2]:.4f}" for k in c), "dm@100", f"{r[SEED]['mass_drift'][-1]:.2e}",
              "curl", f"{r['curl']['curl_over_total']:.2f}" if "curl" in r else "-", "selff", f"{r['self_force']:.3f}")
print("## sweep (D): d/h, err@20, sepErr/d @100, dE/E @100, dL/L @100")
Dr = load("D")
Dr["baseline"] = json.load(open(f"{res}/D_baseline.json"))
for n, r in Dr.items():
    if "sweep" not in r:
        continue
    print(n)
    for k in sorted(r["sweep"], key=float):
        s = r["sweep"][k]
        print(f"  {float(k):6.2f} err@20 {s['err'][2]:.4f} err@100 {s['err'][4]:.3f} sep/d@100 {s['sep_rel_err'][2]:.3f} dE {s['energy_drift_rel']:.3f} dL {s['angmom_drift_rel']:.3f}")
print("## delete-star (C): lag steps / pre-deletion model/truth ratio")
for n, r in load("C").items():
    if "probe" in r:
        p = r["probe"]
        print(n, "lag", p["lag_steps"], "ratio_before", [round(x, 2) for x in p["ratio_before"]])
