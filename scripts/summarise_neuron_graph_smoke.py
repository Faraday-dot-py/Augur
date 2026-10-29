import json
import sys

import numpy as np

ARMS = ["full-synaptic-messages", "no-message-ablation", "per-cell-mlp-no-token-no-messages",
        "linear-dynamics-fit-with-synaptic-input", "persistence-hold-last-state", "teacher-perturbed-0.01mV-floor"]
METRICS = ["err@1", "err@10", "err@20", "err@50", "err@100", "err@300", "spike_f1@20", "spike_f1@100", "spike_time_error@100",
           "cell_rate_error_hz", "divergence_horizon", "unbounded_fraction", "first_nonfinite_step_min"]


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    for r in runs:
        print(f"seed {r['seed']} verdict {r['verdict']} wall {r['wall_seconds']:.0f}s params {r['parameters']}")
        failed = {k: v for k, v in r["verdict_details"].items() if isinstance(v, list) and not v[2]}
        print("  failed comparisons:", failed if failed else "none")
        print("  ablation relative excess:", {k: round(v, 3) for k, v in r["verdict_details"].items() if "excess" in k})
    for n in ("10", "100"):
        print(f"\nN={n}: mean over seeds of (mean over 10 circuits) [seed-spread std / mean circuit-std]")
        print("arm | " + " | ".join(METRICS))
        for arm in ARMS:
            cells = []
            for m in METRICS:
                means = [r["metrics"][n][arm][m][0] for r in runs]
                cstd = [r["metrics"][n][arm][m][1] for r in runs]
                cells.append(f"{np.mean(means):.3f} [{np.std(means):.3f}/{np.mean(cstd):.3f}]")
            print(arm + " | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
