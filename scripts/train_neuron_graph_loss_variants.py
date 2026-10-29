import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.neuron_graph import NeuronGraphDynamics, SPIKE_POS_WEIGHT, count_parameters, normalize_state
from scripts.neuron_loss_diagnostic import full_raw_state, prepare_eval, rate_statistics
from scripts.neuron_circuit_teacher import simulate
from scripts.train_neuron_graph_smoke import (
    EVAL_HORIZON, EVAL_SIZES, SEG_TRAIN, TRAIN_SIZES, build_arm, free_rollout, make_heldout_data, make_train_data,
    rollout_metrics, sample_batch,
)

VARIANTS = ("smoothed-spike-timing-tolerant-van-rossum", "statistics-matching-rate-and-correlation", "gaussian-nll-voltage-head")
SMOOTH_SIGMA = 2.0
SMOOTH_WEIGHT = 100.0
CORR_SIGMA = 5.0
STATS_WINDOW = 64
TRAJECTORY_CAP = SEG_TRAIN


class GaussianVoltageDynamics(NeuronGraphDynamics):
    def __init__(self, messages=True, **kw):
        super().__init__(messages=messages, **kw)
        self.logvar = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)
        self.last_logvar = None

    def step(self, x, ext, h, graph, gate=None):
        nxt, logit, h = super().step(x, ext, h, graph, gate)
        self.last_logvar = self.logvar(h).reshape(x.shape[0], x.shape[1]).clamp(-6.0, 3.0)
        return nxt, logit, h


def gaussian_kernel(sigma, device):
    radius = int(math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return (k / k.sum())[None, None], radius


def smooth(seq, sigma):
    """seq (T, B, N) -> (T, B, N) gaussian-smoothed along time."""
    t, b, n = seq.shape
    kern, radius = gaussian_kernel(sigma, seq.device)
    flat = seq.permute(1, 2, 0).reshape(b * n, 1, t)
    out = F.conv1d(F.pad(flat, (radius, radius), mode="constant"), kern)
    return out.reshape(b, n, t).permute(2, 0, 1)


def variant_loss(model, graph, st, ex, rollout, variant):
    x = st[:, 0]
    h = model.init_hidden(x, ex[:, 0])
    pos = torch.tensor(SPIKE_POS_WEIGHT, device=x.device)
    length = ex.shape[1]
    cap = min(length, TRAJECTORY_CAP) if variant == "statistics-matching-rate-and-correlation" else length
    traj, probs, targets_spk = 0.0, [], []
    for k in range(length):
        target = st[:, k + 1]
        x_next, logit, h = model.step(x, ex[:, k], h, graph, None if rollout else target[..., 3])
        probs.append(torch.sigmoid(logit))
        targets_spk.append(target[..., 3])
        if k < cap:
            bce = F.binary_cross_entropy_with_logits(logit, target[..., 3], pos_weight=pos)
            if variant == "gaussian-nll-voltage-head":
                lv = model.last_logvar
                v_nll = (0.5 * (torch.exp(-lv) * (x_next[..., 0] - target[..., 0]) ** 2 + lv)).mean()
                rest = F.mse_loss(x_next[..., 1:3], target[..., 1:3]) * (2.0 / 3.0)
                traj = traj + v_nll / 3.0 + rest + 0.5 * bce
            elif variant == "smoothed-spike-timing-tolerant-van-rossum":
                traj = traj + F.mse_loss(x_next[..., :3], target[..., :3])
            else:
                traj = traj + F.mse_loss(x_next[..., :3], target[..., :3]) + 0.5 * bce
        x = x_next if rollout else target
    loss = traj / cap
    p, s = torch.stack(probs), torch.stack(targets_spk)
    if variant == "smoothed-spike-timing-tolerant-van-rossum":
        loss = loss + SMOOTH_WEIGHT * 0.5 * F.mse_loss(smooth(p, SMOOTH_SIGMA), smooth(s, SMOOTH_SIGMA))
    if variant == "statistics-matching-rate-and-correlation" and rollout:
        rate_p, rate_s = p.mean(dim=0), s.mean(dim=0)
        rate_term = ((rate_p - rate_s) ** 2).mean() / (rate_s.mean() ** 2 + 1e-4)
        gp, gs = smooth(p, CORR_SIGMA), smooth(s, CORR_SIGMA)
        corr_term = F.mse_loss(correlation_matrix(gp), correlation_matrix(gs))
        loss = loss + rate_term + corr_term
    return loss


def correlation_matrix(trace):
    """trace (T, B, N) -> (B, N, N) correlation across cells within each circuit."""
    z = trace - trace.mean(dim=0, keepdim=True)
    cov = torch.einsum("tbi,tbj->bij", z, z) / trace.shape[0]
    std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2) + 1e-6)
    return cov / (std[:, :, None] * std[:, None, :])


def build_variant_arm(variant, messages):
    if variant == "gaussian-nll-voltage-head":
        return GaussianVoltageDynamics(messages=messages)
    return NeuronGraphDynamics(messages=messages)


def train_variant(variant, messages, data, seed, args, device, log, tag):
    torch.manual_seed(seed)
    model = build_variant_arm(variant, messages).to(device)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    stage2_len = STATS_WINDOW if variant == "statistics-matching-rate-and-correlation" else SEG_TRAIN
    for stage, rollout, iters, length in (("teacher-forced", False, args.stage1_iters, SEG_TRAIN),
                                          ("rollout-in-loop", True, args.stage2_iters, stage2_len)):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - 0.9 * i / max(1, iters))
        run = 0.0
        for it in range(iters):
            n = TRAIN_SIZES[int(torch.randint(0, len(TRAIN_SIZES), (1,), generator=gen))]
            graph, st, ex = sample_batch(data, n, args.batch, length, gen, device)
            loss = variant_loss(model, graph, st, ex, rollout, variant)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run = 0.98 * run + 0.02 * float(loss.detach()) if it else float(loss.detach())
            if (it + 1) % 500 == 0:
                log(f"{tag} {stage} it {it + 1}/{iters} loss {run:.4f} t {time.time() - t0:.0f}s")
    return model


def evaluate_arms(arms, heldout, seed, device):
    out = {}
    for n in EVAL_SIZES:
        circ, idx, truth, ex, graph = prepare_eval(heldout, n, device)
        truth_r = truth[:, 1:]
        x0 = normalize_state(truth[:, 0])
        row = {}
        for name, m in arms.items():
            m.eval()
            pr = full_raw_state(free_rollout(m, x0, ex, graph, EVAL_HORIZON))
            met = rollout_metrics(pr, truth_r, ex.shape[0] // 8, 8)
            row[name] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in met.items()}
            row[name]["per_circuit"] = {k: [float(x) for x in met[k]] for k in ("err@10", "err@20", "spike_f1@20", "divergence_horizon")}
            row[name]["rate_statistics"] = {str(h): rate_statistics(pr, truth_r, h) for h in (100, 300)}
        out[n] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--seed", type=int, default=4738)
    p.add_argument("--device", default="cuda")
    p.add_argument("--baseline-dir", default="results/neuron_graph_smoke/seed4738")
    p.add_argument("--out-dir", default="results/neuron_graph_loss_variants")
    p.add_argument("--stage1-iters", type=int, default=1500)
    p.add_argument("--stage2-iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--circuits-per-size", type=int, default=64)
    p.add_argument("--eval-circuits", type=int, default=10)
    args = p.parse_args()
    device = torch.device(args.device)
    out_dir = os.path.join(args.out_dir, args.variant, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    def log(m):
        print(m, flush=True)

    data = make_train_data(args.seed, device, args.circuits_per_size, 2, 300)
    heldout = make_heldout_data(args.seed, device, args.eval_circuits, EVAL_HORIZON + 200)
    arms = {}
    for name in ("full-synaptic-messages", "no-message-ablation"):
        m = build_arm(name).to(device)
        m.load_state_dict(torch.load(os.path.join(args.baseline_dir, f"{name}.pt"), map_location=device))
        arms[f"baseline-{name}"] = m
    for messages, label in ((True, "full-synaptic-messages"), (False, "no-message-ablation")):
        path = os.path.join(out_dir, f"{label}.pt")
        m = build_variant_arm(args.variant, messages).to(device)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
        else:
            m = train_variant(args.variant, messages, data, args.seed, args, device, log, f"{args.variant}-{label}")
            torch.save(m.state_dict(), path)
        arms[f"variant-{label}"] = m
    res = evaluate_arms(arms, heldout, args.seed, device)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"variant": args.variant, "seed": args.seed, "args": vars(args),
                   "parameters": {k: count_parameters(m) for k, m in arms.items()}, "eval": res}, f, indent=1)
    for n in EVAL_SIZES:
        log(f"N={n}")
        for name, r in res[n].items():
            log(f"  {name}: err@10 {r['err@10'][0]:.2f} err@20 {r['err@20'][0]:.2f} f1@20 {r['spike_f1@20'][0]:.2f} f1@100 {r['spike_f1@100'][0]:.2f} "
                f"div {r['divergence_horizon'][0]:.1f} rate300 ratio {r['rate_statistics']['300']['ratio']:.2f} slope {r['rate_statistics']['300']['slope_pred_on_true']:.2f} corr {r['rate_statistics']['300']['correlation']:.2f}")
    log("DONE")


if __name__ == "__main__":
    main()
