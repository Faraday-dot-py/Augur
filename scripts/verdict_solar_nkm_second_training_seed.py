"""Registered verdict for the solar n/k/m regimes second-training-seed rerun (arms n-and-m-and-k, n-and-k, k-alone, control-all-full at training seeds 4739 and 4740).
Per arm and seed, the same line as job 4918 (docs/debugging/solar-n-k-m-curriculum-regimes-preregistration.md): PASS = two-moons survival >= 1100 and all guards; KILL = survival <= kill line or a guard fails; else INCONCLUSIVE.
Kill line = max(900, 841.6 + floor), floor from the moon arm repeat file. Guards: held-out planets kept mean >= 0.67; original solar kept >= 7 of 8 on seeds 9300 and 12000.
Reads survival from the enlarged eval json and guard numbers from the guard eval json; the eval scripts' own verdict fields are not used.
An arm replicates when its verdict class is the same on every new seed. The 4918 (seed 4738) verdict is shown as context only.

  python scripts/verdict_solar_nkm_second_training_seed.py RESULT_ROOT REPEAT3_JSON REF_VERDICT_JSON OUT_JSON
RESULT_ROOT holds one directory s<seed> per training seed with <arm>_enlarged_eval.json and <arm>_guard_eval/eval_summary.json.
"""
import json
import os
import re
import sys

ARMS = ["n-and-m-and-k", "n-and-k", "k-alone", "control-all-full"]


def classify(surv, held, orig, kill_line):
    fails = int(held < 0.67) + sum(int(x < 7) for x in orig.values())
    return ("PASS" if surv >= 1100 and fails == 0 else "KILL" if surv <= kill_line or fails else "INCONCLUSIVE"), fails


def main(root, rep, ref, out):
    mm = json.load(open(rep))["models"]["moon-arm"]
    floor = max(mm["noise_floor_range"], mm["ci95_half_width"])
    kill_line = max(900.0, 841.6 + floor)
    seeds = sorted(d[1:] for d in os.listdir(root) if re.fullmatch(r"s\d+", d))
    refarms = json.load(open(ref))["arms"]
    res = {"kill_line": kill_line, "noise_floor": floor, "seeds": seeds, "arms": {}}
    for a in ARMS:
        per = {}
        for s in seeds:
            ef, gf = f"{root}/s{s}/{a}_enlarged_eval.json", f"{root}/s{s}/{a}_guard_eval/eval_summary.json"
            if not (os.path.exists(ef) and os.path.exists(gf)):
                per[s] = {"verdict": "MISSING"}
                continue
            m = json.load(open(ef))["models"][a]
            surv, ci = m["mean_of_repeats"], m["repeats"][0]["ci95"]
            v = json.load(open(gf))["verdicts"][a]
            held, orig = v["held_out_mean"], v["original_solar_kept_2000_confirm"]
            verdict, fails = classify(surv, held, orig, kill_line)
            per[s] = {"survival": surv, "ci95": ci, "ci_width": ci[1] - ci[0], "held_out_kept_mean": held, "original_solar_kept_confirm": orig, "guard_failures": fails,
                      "planets_kept_two_moons": m["repeats"][0]["planets_kept_2000"], "verdict": verdict}
        classes = {p["verdict"] for p in per.values()}
        ref_row = refarms.get(a, {})
        res["arms"][a] = {"per_seed": per, "replicates": len(classes) == 1 and "MISSING" not in classes and len(per) >= 2,
                          "verdict_classes": sorted(classes), "mean_survival": sum(p["survival"] for p in per.values() if "survival" in p) / max(1, sum("survival" in p for p in per.values())),
                          "seed_4738_job_4918": {"survival": ref_row.get("survival"), "verdict": ref_row.get("verdict"), "held_out_kept_mean": ref_row.get("held_out_kept_mean")}}
    for s in seeds:
        c = res["arms"]["control-all-full"]["per_seed"][s]
        for a in ARMS:
            p = res["arms"][a]["per_seed"][s]
            if "survival" in p and "survival" in c:
                p["vs_control_ci"] = "above" if p["survival"] > c["ci95"][1] else "below" if p["survival"] < c["ci95"][0] else "inside"
        ranked = sorted((a for a in ARMS if "survival" in res["arms"][a]["per_seed"][s]), key=lambda a: (-res["arms"][a]["per_seed"][s]["survival"], res["arms"][a]["per_seed"][s]["ci_width"], res["arms"][a]["per_seed"][s]["guard_failures"]))
        res.setdefault("ranking_by_seed", {})[s] = ranked
    json.dump(res, open(out, "w"), indent=1)
    print(f"kill line {kill_line:.1f} (floor {floor:.1f})")
    for a in ARMS:
        r = res["arms"][a]
        print(f"{a:18s} replicates {r['replicates']} classes {r['verdict_classes']} mean survival {r['mean_survival']:.1f} (seed 4738: {r['seed_4738_job_4918']['survival']} {r['seed_4738_job_4918']['verdict']})")
        for s, p in r["per_seed"].items():
            if "survival" in p:
                print(f"   seed {s} {p['survival']:7.1f} CI {p['ci95'][0]:.0f}-{p['ci95'][1]:.0f} held {p['held_out_kept_mean']:.3f} orig {p['original_solar_kept_confirm']} {p['verdict']} vs control {p.get('vs_control_ci')}")
            else:
                print(f"   seed {s} MISSING")


if __name__ == "__main__":
    main(*sys.argv[1:5])
