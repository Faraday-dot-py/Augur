import argparse
import json
import os

import numpy as np
import torch

from model.neuron_graph import Graph, normalize_state
from scripts.neuron_loss_diagnostic import prepare_eval, regime_flags
from scripts.train_neuron_graph_smoke import EVAL_HORIZON, EVAL_SIZES, build_arm, make_heldout_data

BINS = ((-200.0, -65.0), (-65.0, -55.0), (-55.0, -45.0), (-45.0, -30.0), (-30.0, 40.0))


def one_step_spike_ticks(model, truth, ex, graph, exc, code, empty_graph=None, warm=8, steps=60, own_gate=False):
    x = normalize_state(truth[:, 0])
    use = empty_graph if empty_graph is not None else graph
    h = model.init_hidden(x, ex[:, 0])
    rows = []
    for k in range(steps):
        target = normalize_state(truth[:, k + 1])
        x_next, logit, h = model.step(x, ex[:, k], h, use, None if own_gate else target[..., 3])
        if k >= warm:
            m = code[:, k] == 0
            err = (x_next[..., 0] - target[..., 0]) * 20.0
            prev_v = truth[:, k, :, 0]
            true_next = truth[:, k + 1, :, 0]
            pred_next = x_next[..., 0] * 20.0 - 65.0
            rows.append(torch.stack([err[m], prev_v[m], true_next[m], pred_next[m], exc.expand_as(err)[m].float(),
                                     torch.full_like(err[m], float(k))], dim=-1))
        x = target
    return torch.cat(rows).cpu().numpy()


def summarise(rows):
    err, prev_v, true_next, pred_next, is_exc, k = rows.T
    sq = err ** 2
    order = np.sort(sq)[::-1]
    out = {"count": int(len(err)), "rmse": float(np.sqrt(sq.mean())), "median_abs": float(np.median(np.abs(err))),
           "p90_abs": float(np.percentile(np.abs(err), 90)), "p99_abs": float(np.percentile(np.abs(err), 99)),
           "bias": float(err.mean()), "share_sq_error_top5pct": float(order[: max(1, len(order) // 20)].sum() / sq.sum()),
           "corr_pred_true_next": float(np.corrcoef(pred_next, true_next)[0, 1]),
           "true_next_mean": float(true_next.mean()), "true_next_std": float(true_next.std()),
           "rmse_exc": float(np.sqrt(sq[is_exc > 0.5].mean())), "rmse_inh": float(np.sqrt(sq[is_exc < 0.5].mean())) if (is_exc < 0.5).any() else None}
    out["by_prev_voltage"] = {}
    for lo, hi in BINS:
        m = (prev_v >= lo) & (prev_v < hi)
        if m.any():
            out["by_prev_voltage"][f"{lo:.0f}..{hi:.0f}"] = {"fraction": float(m.mean()), "rmse": float(np.sqrt(sq[m].mean())), "bias": float(err[m].mean())}
    out["by_window_step"] = {f"{a}-{b}": float(np.sqrt(sq[(k >= a) & (k < b)].mean())) for a, b in ((8, 12), (12, 24), (24, 60)) if ((k >= a) & (k < b)).any()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4738)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default="results/neuron_graph_smoke/seed4738")
    p.add_argument("--out", default="results/neuron_loss_diagnostic/spike_tick_investigation_seed4738.json")
    args = p.parse_args()
    device = torch.device(args.device)
    heldout = make_heldout_data(args.seed, device, 10, EVAL_HORIZON + 200)
    arms = {}
    for name in ("full-synaptic-messages", "no-message-ablation"):
        m = build_arm(name).to(device)
        m.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"{name}.pt"), map_location=device))
        m.eval()
        arms[name] = m
    res = {}
    with torch.no_grad():
        for n in EVAL_SIZES:
            circ, idx, truth, ex, graph = prepare_eval(heldout, n, device)
            exc = circ["exc"][idx]
            code = regime_flags(truth)
            empty = Graph(torch.zeros_like(circ["W"][idx]))
            res[n] = {}
            for name, m in arms.items():
                res[n][name + " true-gate"] = summarise(one_step_spike_ticks(m, truth, ex, graph, exc, code))
                res[n][name + " own-gate"] = summarise(one_step_spike_ticks(m, truth, ex, graph, exc, code, own_gate=True))
            res[n]["full-synaptic-messages messages-removed-at-eval true-gate"] = summarise(
                one_step_spike_ticks(arms["full-synaptic-messages"], truth, ex, graph, exc, code, empty_graph=empty))
            res[n]["in_degree_mean"] = float((circ["W"] != 0).float().sum(dim=2).mean())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for n, d in res.items():
        print("N", n)
        for k, v in d.items():
            if isinstance(v, dict):
                print(" ", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ("by_prev_voltage", "by_window_step")})
                print("     by prev v", {a: (round(b["fraction"], 3), round(b["rmse"], 1), round(b["bias"], 1)) for a, b in v["by_prev_voltage"].items()}, "by step", {a: round(b, 1) for a, b in v["by_window_step"].items()})
    print("DONE")


if __name__ == "__main__":
    main()
