"""Scores a learned-per-body-property-channels stage-1 result json against the lines in
docs/debugging/architecture-push-prereg/learned-per-body-property-channels(-double-iterations).md, independently of the trainer's own verdict.
Usage: python scripts/score_stage1_mass_channels_double_iterations.py RESULT_DIR STAGE_ITERS [--out score.json]
"""
import argparse
import json
import re
import sys
from datetime import datetime

LINES = dict(planet=dict(pass_da=0.02, pass_de=0.03, pass_dE=0.02, kill_da=0.15, kill_dE=0.2),
             disk=dict(pass_da=0.05, pass_de=0.05, pass_dE=0.05, kill_da=0.25, kill_dE=0.3),
             base_pass=1.15, base_kill=1.5, q1000_kill_planet=0.3)
SEEDS = ("9100", "9200", "9300")
TICK = "100"


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def seed_verdict(fin, init):
    kill, miss = [], []
    for kind in ("planet", "disk"):
        L = LINES[kind]
        for q in ("10", "100", "1000"):
            r = fin[kind][q][TICK]
            kda = LINES["q1000_kill_planet"] if (kind == "planet" and q == "1000") else L["kill_da"]
            if not r["finite"] or r["da_over_a"] > kda or r["dE_model"] > L["kill_dE"]:
                kill.append(f"{kind} q{q}")
            elif not (r["da_over_a"] <= L["pass_da"] and r["de"] <= L["pass_de"] and r["dE_model"] <= L["pass_dE"]):
                miss.append(f"{kind} q{q}")
    for k in ("5", "10", "20"):
        ratio = fin["base"][k] / max(init["base"][k], 1e-12)
        if ratio > LINES["base_kill"]:
            kill.append(f"gate err@{k}")
        elif ratio > LINES["base_pass"]:
            miss.append(f"gate err@{k}")
    return ("KILL" if kill else "INCONCLUSIVE" if miss else "PASS"), kill, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("stage_iters", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    log(f"START score {a.result_dir}")
    res = json.load(open(f"{a.result_dir}/massfeat_curriculum.json"))
    out = {"per_seed": {}}
    worst = "PASS"
    for s in SEEDS:
        v, kill, miss = seed_verdict(res["final_by_seed"][s], res["init_by_seed"][s])
        fin = res["final_by_seed"][s]
        out["per_seed"][s] = {"verdict": v, "kill_cells": kill, "miss_cells": miss,
                              "planet_da_over_a_tick100": {q: fin["planet"][q][TICK]["da_over_a"] for q in ("10", "100", "1000")},
                              "disk_da_over_a_tick100": {q: fin["disk"][q][TICK]["da_over_a"] for q in ("10", "100", "1000")},
                              "gate_ratio": {k: fin["base"][k] / res["init_by_seed"][s]["base"][k] for k in ("5", "10", "20")}}
        worst = "KILL" if "KILL" in (worst, v) else "INCONCLUSIVE" if "INCONCLUSIVE" in (worst, v) else "PASS"
    out["raw_verdict_no_guard"] = worst
    out["plateau_falling_in_json"] = res["plateau_falling"]
    out["json_verdict"] = res["verdict"]
    out["registered_verdict"] = "INCONCLUSIVE-undertrained" if res["plateau_falling"] and worst != "PASS" else worst
    total = a.stage_iters * 7
    skips = {}
    n = 0
    for line in open(f"{a.result_dir}/run.log"):
        m = re.match(r"it (\d+) .*step skipped \((\d+) total\)", line)
        if m:
            n = int(m.group(2))
            st = (int(m.group(1)) - 1) // a.stage_iters
            skips[st] = skips.get(st, 0) + 1
    out["skipped_steps"] = {"total": n, "of": total, "fraction": n / total, "by_stage": [skips.get(i, 0) for i in range(7)]}
    out["checks"] = {"by_stage_sum_equals_total": sum(skips.values()) == n}
    out["val_hist_tail_points"] = len(res["val_hist_tail"])
    json.dump(out, open(a.out, "w"), indent=1) if a.out else None
    print(json.dumps(out, indent=1))
    ok = out["registered_verdict"] == res["verdict"]
    log(("DONE" if ok else "FAIL json verdict differs from recomputed") + " score")
    sys.exit(0 if ok else 1)


main()
