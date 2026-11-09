"""Score the width-32 loss-spike validation: per seed, does any epoch's mean training loss exceed SPIKE_N times the median of the preceding WINDOW epochs.
Pre-registration: docs/debugging/experiment-log.md 2026-11-09 width-32-loss-spike-seed-validation.
Usage: python scripts/width32_loss_spike_validation_score.py --dir results/width32_loss_spike_validation [--controls results/single_law_network_sizing]
"""
import argparse
import json

import numpy as np

SPIKE_N = 10.0
WINDOW = 5
SYSTEMATIC_MIN = 2
SEEDS = (4739, 4740, 4741)


def spikes(losses):
    out = []
    for i in range(WINDOW, len(losses)):
        base = float(np.median(losses[i - WINDOW:i]))
        r = losses[i] / base
        if not np.isfinite(r) or r > SPIKE_N:
            out.append({"epoch": i + 1, "loss": losses[i], "preceding_median": base, "ratio": float(r)})
    return out


def max_ratio(losses):
    return max(losses[i] / float(np.median(losses[i - WINDOW:i])) for i in range(WINDOW, len(losses)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--controls", default=None)
    args = ap.parse_args()
    out = {"spike_n": SPIKE_N, "window": WINDOW, "systematic_min": SYSTEMATIC_MIN, "seeds": {}}
    for s in SEEDS:
        d = json.load(open(f"{args.dir}/seed{s}/width32.json"))
        assert d["train_seed"] == s and d["width"] == 32 and len(d["train"]["epoch_loss"]) == 50, s
        l = d["train"]["epoch_loss"]
        resumed = "resumed" in open(f"{args.dir}/seed{s}/run.log").read()
        out["seeds"][str(s)] = {"spikes": spikes(l), "max_epoch_ratio": max_ratio(l), "skipped_steps": d["train"]["skipped_steps"], "resumed": resumed}
    n = sum(1 for v in out["seeds"].values() if v["spikes"])
    out["seeds_with_spike"] = n
    out["label"] = "SYSTEMATIC" if n >= SYSTEMATIC_MIN else "SEED-SPECIFIC" if n == 0 else "INCONCLUSIVE"
    if args.controls:
        out["reference_7256"] = {}
        for w in (4, 8, 16, 32, 64):
            l = json.load(open(f"{args.controls}/width{w}.json"))["train"]["epoch_loss"]
            out["reference_7256"][str(w)] = {"spikes": spikes(l), "max_epoch_ratio": max_ratio(l)}
    json.dump(out, open(f"{args.dir}/spike_verdict.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
