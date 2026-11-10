"""Verdicts for staggered-double-grid-with-quadratic-deposit.
full: ratios candidate / control (pooled = mean over seeds of candidate divided by mean over seeds of control) from staggered_double_grid_eval.py json.
  PASS: err@20 <= 0.75 AND probe_half <= 0.50 AND dE@100 <= 1.10 AND cost <= 2.2.  KILL: err@20 > 0.95 OR dE@100 > 1.5 OR any non-finite rollout.  Else INCONCLUSIVE.
  A non-PASS verdict becomes INCONCLUSIVE-undertrained when the plateau guard (scripts/plateau_guard.py) says UNDERTRAINED.
stage1: reads shift_average_stage1.py json; reports 2-pass and 4-pass ratios; CLOSES the experiment (no full run) only when both exceed 0.95 on err@20 (PROPOSED-BY-WORKER, see prereg).
Usage: python scripts/staggered_double_grid_verdict.py full EVAL.json [--guard '{"status": ...}'] | stage1 STAGE1.json"""
import argparse
import json

import numpy as np

PASS = {"err@20": 0.75, "probe_half": 0.50, "dE@100": 1.10, "cost": 2.2}
KILL_ERR, KILL_DE = 0.95, 1.5


def smoke_fill(d):
    for run in d["seeds"].values():
        for m in ("control", "cand"):
            r = run[m]
            for pre in ("err", "dE"):
                have = sorted(int(k.split("@")[1]) for k in r if k.startswith(pre + "@"))
                for t in (5, 20, 100):
                    r.setdefault(f"{pre}@{t}", r[f"{pre}@{max(x for x in have if x <= t) if any(x <= t for x in have) else have[0]}"])
    return d


def pooled(d, key):
    seeds = list(d["seeds"])
    return float(np.mean([d["seeds"][s]["cand"][key] for s in seeds]) / np.mean([d["seeds"][s]["control"][key] for s in seeds]))


def full(d, guard=None):
    if d.get("smoke"):
        d = smoke_fill(d)
    r = {k: pooled(d, k) for k in ("err@5", "err@20", "err@100", "dE@100", "probe_half", "probe_off")}
    r["cost"] = pooled(d, "sec_per_tick")
    r["finite"] = min(d["seeds"][s]["cand"]["finite"] for s in d["seeds"])
    r["per_seed_err@20"] = {s: d["seeds"][s]["cand"]["err@20"] / d["seeds"][s]["control"]["err@20"] for s in d["seeds"]}
    if r["finite"] < 1 or r["err@20"] > KILL_ERR or r["dE@100"] > KILL_DE:
        v = "KILL"
    elif all(r[k] <= PASS[k] for k in PASS):
        v = "PASS"
    else:
        v = "INCONCLUSIVE"
    if v != "PASS" and guard and guard.get("status") == "UNDERTRAINED":
        v = "INCONCLUSIVE-undertrained"
    r["verdict"] = v
    return r


def stage1(d):
    def val(s, k, m):
        r = d["seeds"][s]
        return r[m][k] if m in ("probe_half", "probe_off", "sec_per_tick") else r[k][m]

    out = {}
    for k in ("2", "4"):
        out[k] = {m: float(np.mean([val(s, k, m) for s in d["seeds"]]) / np.mean([val(s, "1", m) for s in d["seeds"]])) for m in ("err@5", "err@20", "err@100", "dE@100", "probe_half", "probe_off", "sec_per_tick")}
    out["closes_experiment"] = bool(out["2"]["err@20"] > KILL_ERR and out["4"]["err@20"] > KILL_ERR)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "stage1"])
    ap.add_argument("path")
    ap.add_argument("--guard", default=None)
    args = ap.parse_args()
    d = json.load(open(args.path))
    print(json.dumps(full(d, json.loads(args.guard) if args.guard else None) if args.mode == "full" else stage1(d), indent=1))


if __name__ == "__main__":
    main()
