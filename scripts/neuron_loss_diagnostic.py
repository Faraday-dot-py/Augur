import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.neuron_graph import Graph, SPIKE_POS_WEIGHT, normalize_state
from scripts.neuron_circuit_teacher import simulate
from scripts.train_neuron_graph_smoke import (
    EVAL_HORIZON, EVAL_SIZES, EVAL_STARTS, KS, NEURAL_ARMS, SEG_TRAIN, TRAIN_SIZES, build_arm, free_rollout,
    make_heldout_data, make_train_data, raw_state, sample_batch,
)

REGIMES = ("spike-tick", "refractory", "near-threshold", "subthreshold")
NEAR_V = -55.0
TERMS = ("voltage", "adaptation", "synaptic", "spike")


def sequence_terms(model, graph, st, ex, rollout):
    """Per-term contributions that sum exactly to train_neuron_graph_smoke.sequence_loss."""
    x = st[:, 0]
    h = model.init_hidden(x, ex[:, 0])
    out = {t: 0.0 for t in TERMS}
    pos = torch.tensor(SPIKE_POS_WEIGHT, device=x.device)
    length = ex.shape[1]
    for k in range(length):
        target = st[:, k + 1]
        x_next, logit, h = model.step(x, ex[:, k], h, graph, None if rollout else target[..., 3])
        for i, name in enumerate(TERMS[:3]):
            out[name] = out[name] + F.mse_loss(x_next[..., i], target[..., i]) / 3.0
        out["spike"] = out["spike"] + 0.5 * F.binary_cross_entropy_with_logits(logit, target[..., 3], pos_weight=pos)
        x = x_next if rollout else target
    return {k: v / length for k, v in out.items()}


def flat_grad(term, params):
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def train_with_logging(name, data, seed, args, device, log):
    torch.manual_seed(seed)
    model = build_arm(name).to(device)
    params = [p for p in model.parameters()]
    gen = torch.Generator().manual_seed(seed)
    records = []
    t_start = time.time()
    it_global = 0
    for stage, rollout, iters in (("teacher-forced", False, args.stage1_iters), ("rollout-in-loop", True, args.stage2_iters)):
        opt = torch.optim.Adam(params, lr=args.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - 0.9 * i / max(1, iters))
        for it in range(iters):
            n = TRAIN_SIZES[int(torch.randint(0, len(TRAIN_SIZES), (1,), generator=gen))]
            graph, st, ex = sample_batch(data, n, args.batch, SEG_TRAIN, gen, device)
            terms = sequence_terms(model, graph, st, ex, rollout)
            loss = sum(terms.values())
            if it % args.log_every == 0:
                gv = {k: flat_grad(v, params) for k, v in terms.items()}
                norms = {k: float(g.norm()) for k, g in gv.items()}
                cos = float(F.cosine_similarity(gv["voltage"], gv["spike"], dim=0))
                records.append({"stage": stage, "iter": it, "n": n, "loss": {k: float(v) for k, v in terms.items()},
                                "grad_norm": norms, "cos_voltage_spike": cos})
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            it_global += 1
            if (it + 1) % 500 == 0:
                log(f"{name} {stage} it {it + 1}/{iters} t {time.time() - t_start:.0f}s")
    return model, records


def summarise_training(records):
    out = {}
    for stage in ("teacher-forced", "rollout-in-loop"):
        rs = [r for r in records if r["stage"] == stage]
        if not rs:
            continue
        n = len(rs)
        thirds = {"first third": rs[: n // 3], "last third": rs[-(n // 3):]}
        out[stage] = {}
        for label, sub in thirds.items():
            loss = {t: float(np.mean([r["loss"][t] for r in sub])) for t in TERMS}
            gn = {t: float(np.mean([r["grad_norm"][t] for r in sub])) for t in TERMS}
            out[stage][label] = {
                "loss": loss, "loss_share": {t: loss[t] / sum(loss.values()) for t in TERMS},
                "grad_norm": gn, "grad_norm_share": {t: gn[t] / sum(gn.values()) for t in TERMS},
                "cos_voltage_spike_mean": float(np.mean([r["cos_voltage_spike"] for r in sub])),
                "cos_voltage_spike_frac_negative": float(np.mean([r["cos_voltage_spike"] < 0 for r in sub]))}
    return out


def regime_flags(truth_full):
    """truth_full (W, T + 1, N, 4) raw. Returns int regime codes (W, T, N) for
    target ticks 1..T from the truth: spike on the target tick; refractory
    (spike within the 3 previous ticks); near-threshold (previous v above
    NEAR_V); else subthreshold."""
    v, spk = truth_full[..., 0], truth_full[..., 3] > 0.5
    t = truth_full.shape[1] - 1
    spike = spk[:, 1:]
    padded = torch.cat([torch.zeros_like(spk[:, :3]), spk], dim=1)
    recent = padded[:, 2:2 + t] | padded[:, 1:1 + t] | padded[:, 0:t]
    near = v[:, :-1] > NEAR_V
    code = torch.full_like(spike, 3, dtype=torch.long)
    code = torch.where(near, torch.full_like(code, 2), code)
    code = torch.where(recent, torch.full_like(code, 1), code)
    code = torch.where(spike, torch.zeros_like(code), code)
    return code


def prepare_eval(heldout, n, device):
    circ, ext, states, _ = heldout[n]
    circuits = ext.shape[0]
    starts = torch.linspace(0, ext.shape[1] - EVAL_HORIZON - 1, EVAL_STARTS).long().tolist()
    idx = torch.arange(circuits, device=device).repeat_interleave(EVAL_STARTS)
    t0 = torch.tensor(starts * circuits, device=device)
    win = t0[:, None] + torch.arange(EVAL_HORIZON + 1, device=device)[None]
    truth = states[idx[:, None], win]
    ex = ext[idx[:, None], win[:, :-1]]
    graph = Graph(circ["W"][idx])
    return circ, idx, truth, ex, graph


def channel_errors(pred, truth):
    """pred, truth (W, T, N, 4) raw. RMSE per channel at each k in KS plus
    spike mismatch fraction."""
    res = {}
    for k in KS:
        d = pred[:, k - 1] - truth[:, k - 1]
        res[k] = {"voltage_rmse_mv": float(d[..., 0].pow(2).mean().sqrt()), "adaptation_rmse": float(d[..., 1].pow(2).mean().sqrt()),
                  "synaptic_rmse": float(d[..., 2].pow(2).mean().sqrt()),
                  "spike_mismatch_frac": float(((pred[:, k - 1, :, 3] > 0.5) != (truth[:, k - 1, :, 3] > 0.5)).float().mean()),
                  "true_spike_frac": float((truth[:, k - 1, :, 3] > 0.5).float().mean()),
                  "pred_spike_frac": float((pred[:, k - 1, :, 3] > 0.5).float().mean())}
    return res


def regime_rollout_errors(pred, truth_r, code):
    res = {}
    for k in (1, 10, 20, 50, 100):
        res[k] = {}
        c = code[:, k - 1]
        sq = (pred[:, k - 1, :, 0] - truth_r[:, k - 1, :, 0]) ** 2
        total = float(sq.sum())
        for i, name in enumerate(REGIMES):
            m = c == i
            res[k][name] = {"cell_step_fraction": float(m.float().mean()), "voltage_rmse_mv": float(sq[m].mean().sqrt()) if m.any() else None,
                            "share_of_voltage_sq_error": float(sq[m].sum()) / max(total, 1e-12)}
    return res


@torch.no_grad()
def teacher_forced_regimes(model, truth, ex, graph, code, warm=8):
    """One-step predictions with the true state fed each tick (token recurrent);
    steps before `warm` are skipped so the token is warmed."""
    x = normalize_state(truth[:, 0])
    h = model.init_hidden(x, ex[:, 0])
    sq_v, bce, prob, true_spk, codes = [], [], [], [], []
    for k in range(60):
        target = normalize_state(truth[:, k + 1])
        x_next, logit, h = model.step(x, ex[:, k], h, graph, target[..., 3])
        if k >= warm:
            sq_v.append((x_next[..., 0] - target[..., 0]) ** 2 * 400.0)
            bce.append(F.binary_cross_entropy_with_logits(logit, target[..., 3], reduction="none"))
            prob.append(torch.sigmoid(logit))
            true_spk.append(target[..., 3])
            codes.append(code[:, k])
        x = target
    sq_v, bce, prob, true_spk, codes = (torch.stack(a) for a in (sq_v, bce, prob, true_spk, codes))
    out = {}
    total_sq, total_bce = float(sq_v.sum()), float(bce.sum())
    for i, name in enumerate(REGIMES):
        m = codes == i
        if not m.any():
            continue
        out[name] = {"cell_step_fraction": float(m.float().mean()), "voltage_rmse_mv_one_step": float(sq_v[m].mean().sqrt()),
                     "share_of_voltage_sq_error": float(sq_v[m].sum()) / total_sq, "bce_per_cell_step": float(bce[m].mean()),
                     "share_of_bce": float(bce[m].sum()) / total_bce, "mean_predicted_spike_prob": float(prob[m].mean()),
                     "true_spike_rate": float(true_spk[m].mean())}
    return out


def regime_gradient_conflict(model, truth, ex, graph, code, warm=8, length=16):
    """Gradient cosine between the voltage-MSE and spike-BCE terms of the
    one-step loss, over all cell-steps and restricted to each regime."""
    params = [p for p in model.parameters()]
    res = {}
    x0 = normalize_state(truth[:, warm])
    for label, regimes in (("all", None), ("near-threshold+spike-tick+refractory", (0, 1, 2)), ("subthreshold", (3,))):
        x = x0
        h = model.init_hidden(x, ex[:, warm])
        lv, lb = 0.0, 0.0
        for k in range(warm, warm + length):
            target = normalize_state(truth[:, k + 1])
            x_next, logit, h = model.step(x, ex[:, k], h, graph, target[..., 3])
            c = code[:, k]
            m = torch.ones_like(c, dtype=torch.bool) if regimes is None else sum((c == r) for r in regimes).bool()
            denom = m.sum().clamp(min=1)
            lv = lv + (((x_next[..., 0] - target[..., 0]) ** 2) * m).sum() / denom
            lb = lb + (F.binary_cross_entropy_with_logits(logit, target[..., 3], reduction="none",
                                                           pos_weight=torch.tensor(SPIKE_POS_WEIGHT, device=x.device)) * m).sum() / denom
            x = target
        gv, gb = flat_grad(lv, params), flat_grad(lb, params)
        res[label] = {"cosine": float(F.cosine_similarity(gv, gb, dim=0)), "grad_norm_voltage": float(gv.norm()), "grad_norm_spike": float(gb.norm())}
    return res


def rate_statistics(pred, truth_r, horizon):
    rt = (truth_r[:, :horizon, :, 3] > 0.5).float().mean(dim=1) * 1000.0
    rp = (pred[:, :horizon, :, 3] > 0.5).float().mean(dim=1) * 1000.0
    rt_f, rp_f = rt.reshape(-1), rp.reshape(-1)
    slope = float(((rt_f - rt_f.mean()) * (rp_f - rp_f.mean())).sum() / ((rt_f - rt_f.mean()) ** 2).sum().clamp(min=1e-9))
    corr = float(torch.corrcoef(torch.stack([rt_f, rp_f]))[0, 1]) if rp_f.std() > 0 else 0.0
    return {"mean_true_hz": float(rt_f.mean()), "mean_pred_hz": float(rp_f.mean()), "ratio": float(rp_f.mean() / rt_f.mean().clamp(min=1e-9)),
            "slope_pred_on_true": slope, "correlation": corr, "silent_cells_true": float((rt_f == 0).float().mean()),
            "silent_cells_pred": float((rp_f == 0).float().mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4738)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-dir", default="results/neuron_graph_smoke/seed4738")
    p.add_argument("--out", default="results/neuron_loss_diagnostic/seed4738.json")
    p.add_argument("--stage1-iters", type=int, default=1500)
    p.add_argument("--stage2-iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--train-arms", nargs="*", default=["full-synaptic-messages", "no-message-ablation"])
    p.add_argument("--skip-training-log", action="store_true")
    p.add_argument("--circuits-per-size", type=int, default=64)
    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def log(m):
        print(m, flush=True)

    result = {"seed": args.seed}
    if not args.skip_training_log:
        data = make_train_data(args.seed, device, args.circuits_per_size, 2, 300)
        result["training"] = {}
        for name in args.train_arms:
            model, records = train_with_logging(name, data, args.seed, args, device, log)
            result["training"][name] = summarise_training(records)
            result["training"][name]["records"] = records
        del data
    heldout = make_heldout_data(args.seed, device, 10, EVAL_HORIZON + 200)
    arms = {}
    for name in NEURAL_ARMS:
        m = build_arm(name).to(device)
        m.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"{name}.pt"), map_location=device))
        m.eval()
        arms[name] = m
    result["eval"] = {}
    for n in EVAL_SIZES:
        circ, idx, truth, ex, graph = prepare_eval(heldout, n, device)
        truth_r = truth[:, 1:]
        code = regime_flags(truth)
        x0 = normalize_state(truth[:, 0])
        row = {"regime_fractions_all_ticks": {nm: float((code == i).float().mean()) for i, nm in enumerate(REGIMES)}}
        preds = {name: raw_state(free_rollout(m, x0, ex, graph, EVAL_HORIZON)) for name, m in arms.items()}
        cw = {k: v[idx] for k, v in circ.items()}
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        s0 = truth[:, 0].clone()
        s0[..., 0] = s0[..., 0] + 0.01 * torch.randn(s0[..., 0].shape, generator=g).to(device)
        preds["teacher-perturbed-0.01mV-floor"] = simulate(cw, ex, s0)[:, 1:]
        row["channel_errors_vs_step"] = {name: channel_errors(pr, truth_r) for name, pr in preds.items()}
        row["rollout_error_by_regime"] = {name: regime_rollout_errors(pr, truth_r, code) for name, pr in preds.items()}
        row["rate_statistics"] = {name: {h: rate_statistics(pr, truth_r, h) for h in (100, 300)} for name, pr in preds.items()}
        row["teacher_forced_one_step_by_regime"] = {name: teacher_forced_regimes(m, truth, ex, graph, code) for name, m in arms.items()
                                                    if not name.startswith("per-cell")}
        row["gradient_conflict_trained_checkpoint"] = {name: regime_gradient_conflict(m, truth, ex, graph, code) for name, m in arms.items()
                                                       if not name.startswith("per-cell")}
        result["eval"][n] = row
        log(f"N={n} eval done")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
