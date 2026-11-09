import argparse
import json
import math
import os
import sys

import numpy as np

SEEDS = tuple(range(4738, 4748))
UNSEEN = tuple(range(4741, 4748))
PRIOR = {
    "slow-patch": {4738: "PARTIAL", 4739: "PARTIAL", 4740: "KILL"},
    "plasticity": {4738: "KILL", 4739: "PARTIAL", 4740: "PARTIAL"},
}
MIN_SEEDS = 8
ARMS = ("baseline-full-synaptic-messages", "control-baseline-architecture-48-tick-segments-control", "variant-full-synaptic-messages")
PLASTIC, FIXED = "plastic-synapse-state", "fixed-synapse-no-mutable-state"


def load(root, idea, seed):
    path = os.path.join(root, idea, f"seed{seed}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate(verdicts, n_total, min_seeds=MIN_SEEDS):
    n = n_total
    if len(verdicts) < min_seeds:
        return "INCOMPLETE", n
    need = math.ceil(0.6 * n)
    npass = sum(v == "PASS" for v in verdicts)
    nkill = sum(v == "KILL" for v in verdicts)
    if npass >= need:
        return "PASS", n
    return ("FAIL-KILL" if nkill >= math.ceil(0.5 * n) else "FAIL-PARTIAL"), n


def slow_extra(runs):
    for n in ("10", "100"):
        for key, pick in (("err@20", lambda r: r["err@20"][0]), ("f1@20", lambda r: r["spike_f1@20"][0]),
                          ("slope300", lambda r: r["rate_statistics"]["300"]["slope_pred_on_true"])):
            diff_bl = np.array([pick(d["eval"][n][ARMS[2]]) - pick(d["eval"][n][ARMS[0]]) for d in runs.values()])
            diff_ct = np.array([pick(d["eval"][n][ARMS[2]]) - pick(d["eval"][n][ARMS[1]]) for d in runs.values()])
            base = np.array([pick(d["eval"][n][ARMS[0]]) for d in runs.values()])
            print(f"  N={n} {key}: baseline seed sd {base.std(ddof=1):.3f}; variant-baseline mean {diff_bl.mean():+.3f} sd {diff_bl.std(ddof=1):.3f}; "
                  f"variant-control mean {diff_ct.mean():+.3f} sd {diff_ct.std(ddof=1):.3f} (reported, not gating)")
    works = sum(all(d["verdict_details"][n]["checks"][k] for n in ("10", "100") for k in ("slope300>=0.3", "corr300>=0.3")) for d in runs.values())
    print(f"  seeds with rate signal at both N (slope300>=0.3 and corr300>=0.3): {works}/{len(runs)} (reported, not gating)")


def plastic_extra(runs):
    for n in ("10", "100"):
        for k in (0, 1):
            cp = np.array([np.mean(np.asarray(d["episodes"][n]["forward"][PLASTIC]["dR"])[:, k]) - np.mean(np.asarray(d["episodes"][n]["reversed"][PLASTIC]["dR"])[:, k]) for d in runs.values()])
            cf = np.array([np.mean(np.asarray(d["episodes"][n]["forward"][FIXED]["dR"])[:, k]) - np.mean(np.asarray(d["episodes"][n]["reversed"][FIXED]["dR"])[:, k]) for d in runs.values()])
            print(f"  N={n} probe{k + 1}: plastic contrast mean {cp.mean():+.2f} sd {cp.std(ddof=1):.2f} positive {int((cp > 0).sum())}/{len(cp)}; fixed contrast mean {cf.mean():+.2f} sd {cf.std(ddof=1):.2f} (reported, not gating)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/neuron_seed_robust")
    args = p.parse_args()
    missing = 0
    for idea, extra in (("slow-patch", slow_extra), ("plasticity", plastic_extra)):
        runs = {s: d for s in SEEDS if (d := load(args.root, idea, s)) is not None}
        missing += len(SEEDS) - len(runs)
        for s in SEEDS:
            if s in runs:
                prior = PRIOR[idea].get(s)
                note = f" (earlier run: {prior}, same verdict {prior == runs[s]['verdict']})" if prior else ""
                print(f"{idea} seed {s}: {runs[s]['verdict']}{note}")
            else:
                print(f"{idea} seed {s}: MISSING")
        verdicts = [d["verdict"] for d in runs.values()]
        counts = {v: verdicts.count(v) for v in ("PASS", "PARTIAL", "KILL")}
        agg, n = aggregate(verdicts, len(SEEDS))
        print(f"{idea}: seeds completed {len(verdicts)} of {n}, counts {counts}, AGGREGATE {agg}")
        unseen = [d["verdict"] for s, d in runs.items() if s in UNSEEN]
        agg_unseen, _ = aggregate(unseen, len(UNSEEN), 6)
        same = agg_unseen == agg or (agg_unseen.startswith("FAIL") and agg.startswith("FAIL"))
        print(f"{idea}: unseen seeds only (4741-4747) AGGREGATE {agg_unseen}, agrees with all-seed pass/fail class: {same}")
        if len(runs) >= 2:
            extra(runs)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
