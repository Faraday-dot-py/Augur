import argparse
import json
import os
import sys

SEED_4738_VERDICT = {
    "slow-patch-rate-token-with-patch-count-head": "PARTIAL",
    "windowed-causal-temporal-attention-memory": "KILL",
    "plastic-synapse-state-stdp-induction-probe": "KILL",
    "bit-output-cell-mlp": "PASS",
}
SEEDS = (4739, 4740)


def bit_verdict(path):
    with open(path) as f:
        err = json.load(f)["particle_force_rel_err"]
    ratios = {m: err[f"bits-{m}"] / err[f"float-{m}"] for m in ("global", "centered")}
    if any(r > 3.0 for r in ratios.values()):
        return "KILL", ratios
    return ("PASS" if all(r <= 1.5 for r in ratios.values()) else "PARTIAL"), ratios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/neuron_extra_training_seeds")
    args = p.parse_args()
    components = {
        "slow-patch-rate-token-with-patch-count-head": lambda s: os.path.join(args.root, "slow-patch", f"seed{s}", "results.json"),
        "windowed-causal-temporal-attention-memory": lambda s: os.path.join(args.root, "windowed-attention", f"seed{s}", "results.json"),
        "plastic-synapse-state-stdp-induction-probe": lambda s: os.path.join(args.root, "plasticity", f"seed{s}", "results.json"),
        "bit-output-cell-mlp": lambda s: os.path.join(args.root, "bit-output", f"seed{s}.json"),
    }
    missing = 0
    for name, path_of in components.items():
        verdicts = {}
        for s in SEEDS:
            path = path_of(s)
            if not os.path.exists(path):
                print(f"{name} seed {s}: MISSING {path}")
                missing += 1
                continue
            if name == "bit-output-cell-mlp":
                verdicts[s], detail = bit_verdict(path)
            else:
                with open(path) as f:
                    verdicts[s] = json.load(f)["verdict"]
                detail = ""
            print(f"{name} seed {s}: {verdicts[s]} {detail}")
        if len(verdicts) == len(SEEDS):
            same = all(v == SEED_4738_VERDICT[name] for v in verdicts.values())
            print(f"{name}: seed 4738 {SEED_4738_VERDICT[name]}, replicates {same}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
