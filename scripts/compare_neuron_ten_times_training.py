import json
import sys

ARMS = ["full-synaptic-messages", "no-message-ablation", "per-cell-mlp-no-token-no-messages", "linear-dynamics-fit-with-synaptic-input"]
M = ["err@1", "err@10", "err@20", "err@100", "spike_f1@20", "spike_f1@100", "cell_rate_error_hz", "divergence_horizon"]


def main(base, ten):
    b, t = json.load(open(base)), json.load(open(ten))
    print("verdict base", b["verdict"], "10x", t["verdict"], "wall", round(b["wall_seconds"]), round(t["wall_seconds"]))
    for n in ("10", "100"):
        print(f"N={n}")
        for a in ARMS:
            print(" ", a, " | ".join(f"{m} {b['metrics'][n][a][m][0]:.2f}->{t['metrics'][n][a][m][0]:.2f}" for m in M))
        f1, f10 = b["metrics"][n]["full-synaptic-messages"], t["metrics"][n]["full-synaptic-messages"]
        a1, a10 = b["metrics"][n]["no-message-ablation"], t["metrics"][n]["no-message-ablation"]
        print("  full err@20 change %", round(100 * (f10["err@20"][0] / f1["err@20"][0] - 1), 1),
              "div horizon change %", round(100 * (f10["divergence_horizon"][0] / f1["divergence_horizon"][0] - 1), 1),
              "ablation gap/full err@20 base", round((a1["err@20"][0] - f1["err@20"][0]) / f1["err@20"][0], 3),
              "10x", round((a10["err@20"][0] - f10["err@20"][0]) / f10["err@20"][0], 3))


if __name__ == "__main__":
    main(*sys.argv[1:])
