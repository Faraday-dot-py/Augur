import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.neuron_graph import (
    Graph, LinearDynamics, NeuronGraphDynamics, PerCellMlpDynamics, PersistenceDynamics, SPIKE_POS_WEIGHT,
    count_parameters, denormalize_voltage, linear_features, normalize_state,
)
from scripts.neuron_circuit_teacher import generate, make_circuit, simulate

TRAIN_SIZES = (10, 20, 40, 70, 100)
EVAL_SIZES = (10, 100)
EVAL_STARTS = 8
EVAL_HORIZON = 300
KS = (1, 5, 10, 20, 50, 100, 200, 300)
TOL = 2
SEG_TRAIN = 16
NEURAL_ARMS = ("full-synaptic-messages", "no-message-ablation", "per-cell-mlp-no-token-no-messages")
HELDOUT_OFFSET = 10_000_000


def build_arm(name):
    if name == "full-synaptic-messages":
        return NeuronGraphDynamics(messages=True)
    if name == "no-message-ablation":
        return NeuronGraphDynamics(messages=False)
    return PerCellMlpDynamics()


def make_train_data(seed, device, circuits_per_size, drives_per_circuit, steps):
    data = {}
    for n in TRAIN_SIZES:
        base = seed * 100_000 + n * 1000
        cseeds = [base + i for i in range(circuits_per_size) for _ in range(drives_per_circuit)]
        dseeds = [base + 500_000 + i for i in range(len(cseeds))]
        circ, ext, states = generate(n, cseeds, dseeds, steps, 100, device)
        data[n] = (circ["W"], ext, normalize_state(states))
    return data


def make_heldout_data(seed, device, circuits, steps):
    data = {}
    for n in EVAL_SIZES:
        base = HELDOUT_OFFSET + seed * 1000 + n * 10
        cseeds = [base + i for i in range(circuits)]
        dseeds = [base + 5000 + i for i in range(circuits)]
        circ, ext, states = generate(n, cseeds, dseeds, steps, 100, device)
        data[n] = (circ, ext, states, cseeds)
    return data


def sample_batch(data, n, batch, length, gen, device):
    w, ext, states = data[n]
    idx = torch.randint(0, w.shape[0], (batch,), generator=gen).to(device)
    t0 = torch.randint(0, ext.shape[1] - length, (batch,), generator=gen).to(device)
    steps = t0[:, None] + torch.arange(length + 1, device=device)[None]
    rows = idx[:, None]
    st = states[rows, steps]
    ex = ext[rows, steps[:, :-1]]
    return Graph(w[idx]), st, ex


def sequence_loss(model, graph, st, ex, rollout):
    x = st[:, 0]
    h = model.init_hidden(x, ex[:, 0])
    loss = 0.0
    length = ex.shape[1]
    for k in range(length):
        target = st[:, k + 1]
        gate = None if rollout else target[..., 3]
        x_next, logit, h = model.step(x, ex[:, k], h, graph, gate)
        mse = F.mse_loss(x_next[..., :3], target[..., :3])
        bce = F.binary_cross_entropy_with_logits(
            logit, target[..., 3], pos_weight=torch.tensor(SPIKE_POS_WEIGHT, device=logit.device))
        loss = loss + mse + 0.5 * bce
        x = x_next if rollout else target
    return loss / length


def train_arm(name, data, seed, args, device, log):
    torch.manual_seed(seed)
    model = build_arm(name).to(device)
    gen = torch.Generator().manual_seed(seed)
    t_start = time.time()
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
                log(f"{name} seed {seed} {stage} it {it + 1}/{iters} loss {run:.4f} t {time.time() - t_start:.0f}s")
    return model


def fit_linear(data, device):
    feats, targets = [], []
    for n, (w, ext, states) in data.items():
        graph = Graph(w)
        for t in range(0, ext.shape[1], 3):
            feats.append(linear_features(states[:, t], ext[:, t], graph).reshape(-1, 8))
            targets.append(states[:, t + 1].reshape(-1, 4))
    model = LinearDynamics()
    model.fit(torch.cat(feats), torch.cat(targets))
    return model


@torch.no_grad()
def free_rollout(model, x0, ext, graph, steps):
    x = x0
    h = model.init_hidden(x, ext[:, 0])
    out = []
    for k in range(steps):
        x, _, h = model.step(x, ext[:, k], h, graph)
        out.append(x)
    return torch.stack(out, dim=1)


def dilate(spikes, radius):
    b, t, n = spikes.shape
    flat = spikes.permute(0, 2, 1).reshape(b * n, 1, t).float()
    out = F.max_pool1d(flat, 2 * radius + 1, stride=1, padding=radius)
    return out.reshape(b, n, t).permute(0, 2, 1) > 0


def spike_f1(true_spk, pred_spk, circuits, starts):
    tp_prec = (pred_spk & dilate(true_spk, TOL)).float().sum(dim=(1, 2))
    tp_rec = (true_spk & dilate(pred_spk, TOL)).float().sum(dim=(1, 2))
    n_pred = pred_spk.float().sum(dim=(1, 2))
    n_true = true_spk.float().sum(dim=(1, 2))
    f1s = []
    for c in range(circuits):
        sl = slice(c * starts, (c + 1) * starts)
        np_, nt = n_pred[sl].sum(), n_true[sl].sum()
        if np_ == 0 and nt == 0:
            f1s.append(1.0)
            continue
        prec = float(tp_prec[sl].sum() / np_) if np_ > 0 else 0.0
        rec = float(tp_rec[sl].sum() / nt) if nt > 0 else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return np.array(f1s)


def spike_time_error(true_spk, pred_spk, circuits, starts, cap=10):
    err = torch.full(true_spk.shape, float(cap), device=true_spk.device)
    for d in range(cap - 1, -1, -1):
        err = torch.where(dilate(pred_spk, d), torch.full_like(err, float(d)), err)
    total = (err * true_spk).sum(dim=(1, 2))
    count = true_spk.float().sum(dim=(1, 2))
    out = []
    for c in range(circuits):
        sl = slice(c * starts, (c + 1) * starts)
        out.append(float(total[sl].sum() / count[sl].sum().clamp(min=1)))
    return np.array(out)


def per_circuit(values, circuits, starts):
    return values.reshape(circuits, starts, *values.shape[1:])


def rollout_metrics(pred, truth, circuits, starts):
    """pred, truth: (circuits * starts, steps, N, 4) raw units (v in mV),
    spike channel 0/1. Returns dict of per-circuit arrays."""
    steps = truth.shape[1]
    nonfinite_any = ~torch.isfinite(pred).flatten(2).all(dim=-1)
    first_bad = torch.where(nonfinite_any.any(dim=1), nonfinite_any.float().argmax(dim=1) + 1, torch.full((pred.shape[0],), steps + 1, device=pred.device))
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)
    vp = pred[..., 0].clamp(-1000.0, 1000.0)
    vt = truth[..., 0]
    sq = (vp - vt) ** 2
    res = {}
    sq_c = per_circuit(sq, circuits, starts)
    for k in [k for k in KS if k <= steps]:
        res[f"err@{k}"] = sq_c[:, :, k - 1].mean(dim=(1, 2)).sqrt().cpu().numpy()
    rmse_t = sq.mean(dim=2).sqrt()
    smooth = F.avg_pool1d(rmse_t[:, None], 5, stride=1, padding=2, count_include_pad=False)[:, 0]
    over = smooth > 10.0
    horizon = torch.where(over.any(dim=1), over.float().argmax(dim=1) + 1, torch.full_like(first_bad, steps))
    res["divergence_horizon"] = per_circuit(horizon.float(), circuits, starts).mean(dim=1).cpu().numpy()
    res["first_nonfinite_step_min"] = per_circuit(first_bad.float(), circuits, starts).amin(dim=1).cpu().numpy()
    res["nonfinite_fraction"] = per_circuit((first_bad <= steps).float(), circuits, starts).mean(dim=1).cpu().numpy()
    unbounded = (vp.amax(dim=(1, 2)) > 60.0) | (vp.amin(dim=(1, 2)) < -130.0)
    res["unbounded_fraction"] = per_circuit(unbounded.float(), circuits, starts).mean(dim=1).cpu().numpy()
    ts, ps = truth[..., 3] > 0.5, pred[..., 3] > 0.5
    for h in (20, 100, steps):
        res[f"spike_f1@{h}"] = spike_f1(ts[:, :h], ps[:, :h], circuits, starts)
    res["spike_time_error@100"] = spike_time_error(ts[:, :100], ps[:, :100], circuits, starts)
    rate_t, rate_p = ts.float().mean(dim=1) * 1000.0, ps.float().mean(dim=1) * 1000.0
    res["cell_rate_error_hz"] = per_circuit((rate_t - rate_p).abs().mean(dim=1), circuits, starts).mean(dim=1).cpu().numpy()
    pop_t, pop_p = rate_t.mean(dim=1), rate_p.mean(dim=1)
    res["population_rate_rel_error"] = per_circuit((pop_p - pop_t).abs() / pop_t.clamp(min=1e-3), circuits, starts).mean(dim=1).cpu().numpy()
    return res


def raw_state(xn):
    return torch.stack([denormalize_voltage(xn[..., 0]), xn[..., 1], xn[..., 2], xn[..., 3]], dim=-1)


def evaluate(arms, heldout, seed, device, dump_dir):
    results = {}
    for n, (circ, ext, states, cseeds) in heldout.items():
        circuits = ext.shape[0]
        starts = torch.linspace(0, ext.shape[1] - EVAL_HORIZON - 1, EVAL_STARTS).long().tolist()
        idx = torch.arange(circuits, device=device).repeat_interleave(EVAL_STARTS)
        t0 = torch.tensor(starts * circuits, device=device)
        win = t0[:, None] + torch.arange(EVAL_HORIZON + 1, device=device)[None]
        truth = states[idx[:, None], win]
        ex = ext[idx[:, None], win[:, :-1]]
        weights = circ["W"][idx]
        graph = Graph(weights)
        x0 = normalize_state(truth[:, 0])
        truth_r = truth[:, 1:]
        results[n] = {}
        preds = {}
        for name, model in arms.items():
            if hasattr(model, "eval"):
                model.eval()
            pn = free_rollout(model, x0, ex, graph, EVAL_HORIZON)
            preds[name] = raw_state(pn)
        g = torch.Generator(device="cpu").manual_seed(seed)
        floor_state = states[idx[:, None], win][:, 0].clone()
        floor_state[..., 0] = floor_state[..., 0] + 0.01 * torch.randn(floor_state[..., 0].shape, generator=g).to(device)
        cw = {k: v[idx] for k, v in circ.items()}
        preds["teacher-perturbed-0.01mV-floor"] = simulate(cw, ex, floor_state)[:, 1:]
        for name, pred in preds.items():
            results[n][name] = rollout_metrics(pred, truth_r, circuits, EVAL_STARTS)
        if dump_dir is not None and "full-synaptic-messages" in preds:
            np.savez_compressed(
                os.path.join(dump_dir, f"rollout_n{n}_seed{seed}.npz"),
                truth=truth_r[:EVAL_STARTS].cpu().numpy(), pred=preds["full-synaptic-messages"][:EVAL_STARTS].cpu().numpy(),
                ablation=preds["no-message-ablation"][:EVAL_STARTS].cpu().numpy(),
                weights=weights[0].cpu().numpy(), exc=circ["exc"][0].cpu().numpy(), ext=ex[:EVAL_STARTS].cpu().numpy(),
                circuit_seed=cseeds[0])
    return results


def summarize(results):
    out = {}
    for n, arms in results.items():
        out[n] = {a: {m: [float(np.mean(v)), float(np.std(v))] for m, v in ms.items()} for a, ms in arms.items()}
    return out


def paired(results, n, a, b, metric):
    d = np.asarray(results[n][a][metric]) - np.asarray(results[n][b][metric])
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def verdict(results):
    full, abl = "full-synaptic-messages", "no-message-ablation"
    others = [abl, "per-cell-mlp-no-token-no-messages", "linear-dynamics-fit-with-synaptic-input"]
    lower, higher = ("err@10", "err@20"), ("spike_f1@20",)
    passes, details = True, {}
    kill_both = True
    for n in EVAL_SIZES:
        r = results[n]
        for m in lower + higher:
            for o in others:
                diff, se = paired(results, n, full, o, m)
                good = (diff < -se) if m in lower else (diff > se)
                details[f"N={n} {m} full vs {o}"] = [diff, se, bool(good)]
                passes = passes and bool(good)
        for m in lower:
            rel = (np.mean(r[abl][m]) - np.mean(r[full][m])) / max(np.mean(r[full][m]), 1e-9)
            details[f"N={n} {m} ablation relative excess"] = float(rel)
            kill_both = kill_both and rel < 0.05
        stable = np.max(r[full]["nonfinite_fraction"]) == 0 and np.mean(r[full]["unbounded_fraction"]) < 0.05
        details[f"N={n} full stable"] = bool(stable)
        passes = passes and bool(stable)
    if kill_both:
        return "KILL", details
    return ("PASS" if passes else "PARTIAL"), details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4738)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/neuron_graph_smoke")
    p.add_argument("--stage1-iters", type=int, default=1500)
    p.add_argument("--stage2-iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--circuits-per-size", type=int, default=64)
    p.add_argument("--drives-per-circuit", type=int, default=2)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--eval-circuits", type=int, default=10)
    args = p.parse_args()
    device = torch.device(args.device)
    out = os.path.join(args.out_dir, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    data = make_train_data(args.seed, device, args.circuits_per_size, args.drives_per_circuit, args.train_steps)
    heldout = make_heldout_data(args.seed, device, args.eval_circuits, EVAL_HORIZON + 200)
    train_seeds = {args.seed * 100_000 + n * 1000 + i for n in TRAIN_SIZES for i in range(args.circuits_per_size)}
    held_seeds = {s for v in heldout.values() for s in v[3]}
    assert not train_seeds & held_seeds
    log(f"data ready {time.time() - t0:.0f}s")

    arms = {}
    for name in NEURAL_ARMS:
        path = os.path.join(out, f"{name}.pt")
        model = build_arm(name).to(device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            log(f"{name} loaded from {path}")
        else:
            model = train_arm(name, data, args.seed, args, device, log)
            torch.save(model.state_dict(), path)
        arms[name] = model
    arms["linear-dynamics-fit-with-synaptic-input"] = fit_linear(data, device)
    arms["persistence-hold-last-state"] = PersistenceDynamics()
    params = {n: count_parameters(m) for n, m in arms.items() if hasattr(m, "parameters")}
    log(f"parameters {params}")

    results = evaluate(arms, heldout, args.seed, device, out)
    verdict_name, details = verdict(results)
    summary = {"seed": args.seed, "args": vars(args), "parameters": params, "verdict": verdict_name,
               "verdict_details": details, "metrics": summarize(results), "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for n in EVAL_SIZES:
        log(f"N={n} (mean over {args.eval_circuits} held-out circuits; std in brackets)")
        for name, ms in summary["metrics"][n].items():
            row = " ".join(f"{m} {v[0]:.3f}({v[1]:.3f})" for m, v in ms.items()
                           if m in ("err@1", "err@10", "err@20", "err@100", "spike_f1@20", "spike_f1@100", "cell_rate_error_hz", "divergence_horizon"))
            log(f"  {name}: {row}")
    log(f"VERDICT {verdict_name}")


if __name__ == "__main__":
    main()
