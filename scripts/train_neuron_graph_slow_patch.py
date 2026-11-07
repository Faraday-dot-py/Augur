import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.neuron_graph import NeuronGraphDynamics, SPIKE_POS_WEIGHT, count_parameters
from model.neuron_graph_slow_patch import PATCH, SlowPatchTokenDynamics, poisson_count_loss
from scripts.train_neuron_graph_loss_variants import evaluate_arms
from scripts.train_neuron_graph_smoke import EVAL_HORIZON, EVAL_SIZES, TRAIN_SIZES, build_arm, make_heldout_data, make_train_data, sample_batch

SLOW = "slow-patch-rate-token-with-patch-count-head"
CONTROL = "baseline-architecture-48-tick-segments-control"
COUNT_WEIGHT = 0.5
SEG = 48


def build(name):
    return SlowPatchTokenDynamics() if name == SLOW else NeuronGraphDynamics()


def sequence_loss(model, graph, st, ex, rollout, step_only=False):
    x = st[:, 0]
    h = model.init_hidden(x, ex[:, 0])
    pos = torch.tensor(SPIKE_POS_WEIGHT, device=x.device)
    length = ex.shape[1]
    step_loss, count_terms = 0.0, []
    for k in range(length):
        target = st[:, k + 1]
        x_next, logit, h = model.step(x, ex[:, k], h, graph, None if rollout else target[..., 3])
        bce = F.binary_cross_entropy_with_logits(logit, target[..., 3], pos_weight=pos)
        step_loss = step_loss + F.mse_loss(x_next[..., :3], target[..., :3]) + 0.5 * bce
        pred = getattr(model, "count_pred", None)
        if pred is not None and k + 1 + PATCH <= length:
            count = st[:, k + 2:k + 2 + PATCH, :, 3].sum(dim=1).reshape(-1)
            count_terms.append(poisson_count_loss(pred, count))
        x = x_next if rollout else target
    loss = step_loss / length
    if count_terms and not step_only:
        loss = loss + COUNT_WEIGHT * torch.stack(count_terms).mean()
    return loss


def train(name, data, seed, args, device, log):
    torch.manual_seed(seed)
    model = build(name).to(device)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for stage, rollout, iters in (("teacher-forced", False, args.stage1_iters), ("rollout-in-loop", True, args.stage2_iters)):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - 0.9 * i / max(1, iters))
        run = 0.0
        for it in range(iters):
            n = TRAIN_SIZES[int(torch.randint(0, len(TRAIN_SIZES), (1,), generator=gen))]
            graph, st, ex = sample_batch(data, n, args.batch, SEG, gen, device)
            loss = sequence_loss(model, graph, st, ex, rollout)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run = 0.98 * run + 0.02 * float(loss.detach()) if it else float(loss.detach())
            if (it + 1) % 250 == 0:
                log(f"{name} {stage} it {it + 1}/{iters} loss {run:.4f} t {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def step_only_loss(model, data, seed, device, batches=20, batch=32):
    gen = torch.Generator().manual_seed(seed + 1)
    out = {}
    for rollout in (False, True):
        vals = []
        for i in range(batches):
            n = TRAIN_SIZES[i % len(TRAIN_SIZES)]
            graph, st, ex = sample_batch(data, n, batch, SEG, gen, device)
            vals.append(float(sequence_loss(model, graph, st, ex, rollout, step_only=True)))
        out["rollout-in-loop" if rollout else "teacher-forced"] = sum(vals) / len(vals)
    return out


def score(res):
    """Pre-registered lines (supervisor-log 2026-11-03), slow-patch arm vs the frozen baseline, both N."""
    out, passes, killed = {}, True, False
    for n in EVAL_SIZES:
        b, v = res[n]["baseline-full-synaptic-messages"], res[n]["variant-full-synaptic-messages"]
        r300 = v["rate_statistics"]["300"]
        checks = {
            "slope300>=0.3": r300["slope_pred_on_true"] >= 0.3,
            "corr300>=0.3": r300["correlation"] >= 0.3,
            "rate_ratio300_in_0.7_1.3": 0.7 <= r300["ratio"] <= 1.3,
            "err@20_within_+3pct": v["err@20"][0] <= 1.03 * b["err@20"][0],
            "f1@20_within_0.03": v["spike_f1@20"][0] >= b["spike_f1@20"][0] - 0.03,
        }
        out[n] = {"values": {"slope300": r300["slope_pred_on_true"], "corr300": r300["correlation"], "ratio300": r300["ratio"],
                             "err@20": v["err@20"][0], "f1@20": v["spike_f1@20"][0]}, "checks": {k: bool(c) for k, c in checks.items()}}
        passes = passes and all(checks.values())
        killed = killed or r300["slope_pred_on_true"] < 0.2 or r300["ratio"] < 0.5
    return ("KILL" if killed else "PASS" if passes else "PARTIAL"), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4738)
    p.add_argument("--device", default="cuda")
    p.add_argument("--baseline-dir", default="results/neuron_graph_smoke/seed4738")
    p.add_argument("--out-dir", default="results/neuron_graph_slow_patch")
    p.add_argument("--stage1-iters", type=int, default=1500)
    p.add_argument("--stage2-iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--circuits-per-size", type=int, default=64)
    p.add_argument("--eval-circuits", type=int, default=10)
    args = p.parse_args()
    device = torch.device(args.device)
    out_dir = os.path.join(args.out_dir, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    def log(m):
        print(m, flush=True)

    data = make_train_data(args.seed, device, args.circuits_per_size, 2, 300)
    heldout = make_heldout_data(args.seed, device, args.eval_circuits, EVAL_HORIZON + 200)
    arms = {}
    base = build_arm("full-synaptic-messages").to(device)
    base.load_state_dict(torch.load(os.path.join(args.baseline_dir, "full-synaptic-messages.pt"), map_location=device))
    arms["baseline-full-synaptic-messages"] = base
    for name, label in ((CONTROL, f"control-{CONTROL}"), (SLOW, "variant-full-synaptic-messages")):
        path = os.path.join(out_dir, f"{name}.pt")
        m = build(name).to(device)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
        else:
            m = train(name, data, args.seed, args, device, log)
            torch.save(m.state_dict(), path)
        arms[label] = m
    res = evaluate_arms(arms, heldout, args.seed, device, out_dir)
    verdict, details = score(res)
    split = {k: step_only_loss(m, data, args.seed, device) for k, m in arms.items()}
    for k, v in split.items():
        log(f"step-only loss (no count term, not part of the line) {k}: {v}")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"idea": SLOW, "seed": args.seed, "args": vars(args), "verdict": verdict, "verdict_details": details,
                   "step_only_loss": split,
                   "parameters": {k: count_parameters(m) for k, m in arms.items()}, "eval": res}, f, indent=1)
    for n in EVAL_SIZES:
        log(f"N={n}")
        for name, r in res[n].items():
            r3 = r["rate_statistics"]["300"]
            log(f"  {name}: err@20 {r['err@20'][0]:.2f} f1@20 {r['spike_f1@20'][0]:.2f} f1@100 {r['spike_f1@100'][0]:.2f} div {r['divergence_horizon'][0]:.1f} "
                f"rate300 ratio {r3['ratio']:.2f} slope {r3['slope_pred_on_true']:.2f} corr {r3['correlation']:.2f}")
    log(f"VERDICT {verdict}")
    log(json.dumps(details))


if __name__ == "__main__":
    main()
