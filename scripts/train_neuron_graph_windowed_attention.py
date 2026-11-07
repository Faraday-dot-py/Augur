import argparse
import json
import os
import time

import torch

from model.neuron_graph import count_parameters
from model.neuron_graph_windowed_attention import WindowedAttentionDynamics
from scripts.train_neuron_graph_loss_variants import evaluate_arms
from scripts.train_neuron_graph_smoke import (
    EVAL_HORIZON, EVAL_SIZES, SEG_TRAIN, TRAIN_SIZES, build_arm, make_heldout_data, make_train_data, sample_batch, sequence_loss,
)

IDEA = "windowed-causal-temporal-attention-memory"


def train(data, seed, args, device, log):
    torch.manual_seed(seed)
    model = WindowedAttentionDynamics().to(device)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for stage, rollout, iters in (("teacher-forced", False, args.stage1_iters), ("rollout-in-loop", True, args.stage2_iters)):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - 0.9 * i / max(1, iters))
        run = 0.0
        for it in range(iters):
            n = TRAIN_SIZES[int(torch.randint(0, len(TRAIN_SIZES), (1,), generator=gen))]
            graph, st, ex = sample_batch(data, n, args.batch, SEG_TRAIN, gen, device)
            loss = sequence_loss(model, graph, st, ex, rollout)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run = 0.98 * run + 0.02 * float(loss.detach()) if it else float(loss.detach())
            if (it + 1) % 250 == 0:
                log(f"{IDEA} {stage} it {it + 1}/{iters} loss {run:.4f} t {time.time() - t0:.0f}s")
    return model


def score(res):
    """Pre-registered lines (supervisor-log 2026-11-03 idea 3), attention arm vs the frozen baseline, both N."""
    out, passes, killed = {}, True, False
    for n in EVAL_SIZES:
        b, v = res[n]["baseline-full-synaptic-messages"], res[n]["variant-full-synaptic-messages"]
        checks = {"err@20_at_least_5pct_below": v["err@20"][0] <= 0.95 * b["err@20"][0],
                  "horizon_at_least_15pct_longer": v["divergence_horizon"][0] >= 1.15 * b["divergence_horizon"][0]}
        out[n] = {"values": {"horizon": v["divergence_horizon"][0], "err@20": v["err@20"][0], "f1@20": v["spike_f1@20"][0]},
                  "checks": {k: bool(c) for k, c in checks.items()}}
        passes = passes and all(checks.values())
        killed = killed or v["err@20"][0] > 0.97 * b["err@20"][0]
    return ("KILL" if killed else "PASS" if passes else "PARTIAL"), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4738)
    p.add_argument("--device", default="cuda")
    p.add_argument("--baseline-dir", default="results/neuron_graph_smoke/seed4738")
    p.add_argument("--out-dir", default="results/neuron_graph_windowed_attention")
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
    base = build_arm("full-synaptic-messages").to(device)
    base.load_state_dict(torch.load(os.path.join(args.baseline_dir, "full-synaptic-messages.pt"), map_location=device))
    path = os.path.join(out_dir, f"{IDEA}.pt")
    m = WindowedAttentionDynamics().to(device)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=device))
    else:
        m = train(data, args.seed, args, device, log)
        torch.save(m.state_dict(), path)
    arms = {"baseline-full-synaptic-messages": base, "variant-full-synaptic-messages": m}
    res = evaluate_arms(arms, heldout, args.seed, device, out_dir)
    verdict, details = score(res)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"idea": IDEA, "seed": args.seed, "args": vars(args), "verdict": verdict, "verdict_details": details,
                   "parameters": {k: count_parameters(a) for k, a in arms.items()}, "eval": res}, f, indent=1)
    for n in EVAL_SIZES:
        log(f"N={n}")
        for name, r in res[n].items():
            log(f"  {name}: err@1 {r['err@1'][0]:.2f} err@10 {r['err@10'][0]:.2f} err@20 {r['err@20'][0]:.2f} f1@20 {r['spike_f1@20'][0]:.2f} "
                f"f1@100 {r['spike_f1@100'][0]:.2f} div {r['divergence_horizon'][0]:.1f}")
    log(f"VERDICT {verdict}")
    log(json.dumps(details))


if __name__ == "__main__":
    main()
