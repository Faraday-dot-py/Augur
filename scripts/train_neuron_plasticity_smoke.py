import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.neuron_graph import Graph, count_parameters
from model.neuron_graph_plastic import PlasticNeuronGraphDynamics, denormalize_plastic_state, normalize_plastic_state
from scripts.neuron_plasticity_teacher import (
    CONDITIONS, EPISODE_STEPS, PROBE_TICKS, build_ext, make_plastic_circuit, pick_synapse, probe_response, run_episodes,
)

TRAIN_SIZES = (10, 30, 60, 100)
EVAL_SIZES = (10, 100)
HELDOUT_OFFSET = 20_000_000
SEG_ROLLOUT = 32
SEG_TEACHER = 16
M_WEIGHT = 20.0
SPIKE_POS_WEIGHT = 10.0
ARMS = {"plastic-synapse-state": True, "fixed-synapse-no-mutable-state": False}


def random_training_ext(n, circ, seed):
    g = torch.Generator().manual_seed(seed)
    pre, post = pick_synapse(circ, g)
    mag = int(torch.randint(2, 13, (1,), generator=g))
    sign = 1 if float(torch.rand(1, generator=g)) < 0.5 else -1
    delta = 0 if float(torch.rand(1, generator=g)) < 0.25 else sign * mag
    pairs = int(torch.randint(3, 11, (1,), generator=g))
    interval = int(torch.randint(25, 51, (1,), generator=g))
    start = int(torch.randint(30, 121, (1,), generator=g))
    extra = 0.003 if float(torch.rand(1, generator=g)) < 0.5 else 0.0
    return build_ext(n, pre, post, delta, seed, pairs=pairs, interval=interval, start=start, extra_rate=extra)


def make_train_data(seed, device, episodes_per_size):
    data = {}
    for n in TRAIN_SIZES:
        parts = []
        for chunk in range(0, episodes_per_size, 20):
            seeds = [seed * 100_000 + n * 1000 + chunk + i for i in range(min(20, episodes_per_size - chunk))]
            circuits = [make_plastic_circuit(n, s) for s in seeds]
            exts = [random_training_ext(n, c, s + 777) for c, s in zip(circuits, seeds)]
            circ, (states, gh) = run_episodes(circuits, exts, device, seed0=seed)
            parts.append((circ["W"], torch.stack(exts).to(device), normalize_plastic_state(states), gh))
        data[n] = tuple(torch.cat([p[k] for p in parts]) for k in range(4))
    return data


def window(data, n, batch, length, gen, device):
    w, ext, states, gh = data[n]
    idx = torch.randint(0, w.shape[0], (batch,), generator=gen).to(device)
    t0 = torch.randint(0, ext.shape[1] - length, (batch,), generator=gen).to(device)
    steps = t0[:, None] + torch.arange(length + 1, device=device)[None]
    graph = Graph(w[idx])
    gw = gh[idx[:, None], steps]
    bi, di, si = graph.dst // n, graph.dst % n, graph.src % n
    m_true = gw[bi, :, di, si].float().T - 1.0
    return graph, states[idx[:, None], steps], ext[idx[:, None], steps[:, :-1]], m_true


def sequence_loss(model, graph, st, ex, m_true, rollout):
    x = st[:, 0]
    h = model.init_hidden(x, ex[:, 0])
    m = m_true[0] if model.plastic else torch.zeros_like(m_true[0])
    length = ex.shape[1]
    loss = 0.0
    pos = torch.tensor(SPIKE_POS_WEIGHT, device=x.device)
    for k in range(length):
        target = st[:, k + 1]
        x_next, logit, h, m_next = model.step(x, ex[:, k], h, m, graph, None if rollout else target[..., 3])
        mse = F.mse_loss(x_next[..., [0, 1, 2, 4]], target[..., [0, 1, 2, 4]])
        bce = F.binary_cross_entropy_with_logits(logit, target[..., 3], pos_weight=pos)
        loss = loss + mse + 0.5 * bce
        if model.plastic:
            loss = loss + M_WEIGHT * F.mse_loss(m_next, m_true[k + 1])
        x = x_next if rollout else target
        m = m_next if rollout else (m_true[k + 1] if model.plastic else m_next)
    return loss / length


def train_arm(name, data, seed, args, device, log):
    torch.manual_seed(seed)
    model = PlasticNeuronGraphDynamics(plastic=ARMS[name]).to(device)
    gen = torch.Generator().manual_seed(seed)
    t_start = time.time()
    for stage, rollout, iters, length in (("teacher-forced", False, args.stage1_iters, SEG_TEACHER),
                                          ("rollout-in-loop", True, args.stage2_iters, SEG_ROLLOUT)):
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 - 0.9 * i / max(1, iters))
        run = 0.0
        for it in range(iters):
            n = TRAIN_SIZES[int(torch.randint(0, len(TRAIN_SIZES), (1,), generator=gen))]
            graph, st, ex, m_true = window(data, n, args.batch, length, gen, device)
            loss = sequence_loss(model, graph, st, ex, m_true, rollout)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run = 0.98 * run + 0.02 * float(loss.detach()) if it else float(loss.detach())
            if (it + 1) % 500 == 0:
                log(f"{name} seed {seed} {stage} it {it + 1}/{iters} loss {run:.4f} t {time.time() - t_start:.0f}s")
    return model


@torch.no_grad()
def free_rollout(model, state0_n, ext, graph, steps):
    x = state0_n
    h = model.init_hidden(x, ext[:, 0])
    m = torch.zeros(graph.w.shape[0], device=x.device)
    xs, ms = [], []
    for k in range(steps):
        x, _, h, m = model.step(x, ext[:, k], h, m, graph)
        xs.append(x)
        ms.append(m)
    return torch.stack(xs, dim=1), torch.stack(ms, dim=1)


def param_checksum(model):
    return float(sum(p.detach().double().sum() for p in model.parameters()))


def evaluate(models, seed, circuits, device, dump_dir):
    out = {}
    for n in EVAL_SIZES:
        cseeds = [HELDOUT_OFFSET + seed * 1000 + n * 10 + i for i in range(circuits)]
        circs = [make_plastic_circuit(n, s) for s in cseeds]
        picks = [pick_synapse(c, torch.Generator().manual_seed(s)) for c, s in zip(circs, cseeds)]
        out[n] = {}
        for cond, delta in CONDITIONS.items():
            exts = [build_ext(n, pre, post, delta, s) for (pre, post), s in zip(picks, cseeds)]
            circ, (states, gh) = run_episodes(circs, exts, device, seed0=seed)
            ext = torch.stack(exts).to(device)
            graph = Graph(circ["W"])
            x0 = normalize_plastic_state(states[:, 0])
            res = {"teacher": states}
            tgt = [(i, post, pre) for i, (pre, post) in enumerate(picks)]
            row = {}
            row["teacher"] = {"dR": response_changes(states, tgt)}
            row["teacher"]["g_final"] = [float(gh[i, -1, post, pre]) for i, post, pre in tgt]
            for name, model in models.items():
                before = param_checksum(model)
                model.eval()
                xn, ms = free_rollout(model, x0, ext, graph, EPISODE_STEPS)
                raw = torch.cat([states[:, :1], denormalize_plastic_state(xn)], dim=1)
                assert param_checksum(model) == before
                finite = bool(torch.isfinite(raw).all())
                raw = torch.nan_to_num(raw, nan=0.0, posinf=1e3, neginf=-1e3)
                bi, di, si = graph.dst // n, graph.dst % n, graph.src % n
                m_final = []
                for i, post, pre in tgt:
                    e = ((bi == i) & (di == post) & (si == pre)).nonzero()[0, 0]
                    m_final.append(float(ms[e, -1]) + 1.0)
                row[name] = {"dR": response_changes(raw, tgt), "g_final": m_final, "finite": finite}
                res[name] = raw
            out[n][cond] = row
            if dump_dir is not None and n in (10, 100) and cond == "forward":
                np.savez_compressed(
                    os.path.join(dump_dir, f"plasticity_rollout_n{n}_seed{seed}.npz"),
                    truth=res["teacher"][:1].cpu().numpy(), pred=res["plastic-synapse-state"][:1].cpu().numpy(),
                    ablation=res["fixed-synapse-no-mutable-state"][:1].cpu().numpy(), weights=circ["W"][0].cpu().numpy(),
                    exc=circ["exc"][0].cpu().numpy(), ext=ext[:1].cpu().numpy(), pre_post=np.array(picks[0]))
    return out


def response_changes(states, targets):
    r = []
    for i, post, _ in targets:
        v, spk = states[i, :, post, 0], states[i, :, post, 3]
        base = float(probe_response(v, spk, post, PROBE_TICKS[0]))
        r.append([float(probe_response(v, spk, post, tp)) - base for tp in PROBE_TICKS[1:]])
    return r


def stat(values):
    v = np.asarray(values)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def verdict(out):
    P, Fx = "plastic-synapse-state", "fixed-synapse-no-mutable-state"
    details, passes, kill_plastic_contrast, kill_vs_fixed = {}, True, True, True
    for n, conds in out.items():
        for k in (0, 1):
            mean = {(a, c): stat(np.asarray(conds[c][a]["dR"])[:, k]) for a in ("teacher", P, Fx) for c in conds}
            fwd, rev, non = mean[(P, "forward")], mean[(P, "reversed")], mean[(P, "none")]
            contrast_p = fwd[0] - rev[0]
            contrast_f = mean[(Fx, "forward")][0] - mean[(Fx, "reversed")][0]
            checks = {
                "forward positive t>=2": bool(fwd[0] > 0 and fwd[0] / max(fwd[1], 1e-9) >= 2),
                "reversed negative t>=2": bool(rev[0] < 0 and -rev[0] / max(rev[1], 1e-9) >= 2),
                "no-induction small": bool(abs(non[0]) <= 0.25 * abs(fwd[0])),
                "fixed contrast <= 0.25 plastic": bool(contrast_f <= 0.25 * contrast_p),
            }
            if k == 0:
                checks["forward >= 0.5 teacher"] = bool(fwd[0] >= 0.5 * mean[("teacher", "forward")][0])
            else:
                checks["persists >= 0.5 of probe 1"] = bool(
                    fwd[0] >= 0.5 * stat(np.asarray(conds["forward"][P]["dR"])[:, 0])[0])
            checks["finite"] = all(conds[c][P]["finite"] for c in conds)
            details[f"N={n} probe{k + 1}"] = {"checks": checks, "plastic forward/reversed/none": [fwd, rev, non],
                                              "teacher forward/reversed/none": [mean[("teacher", c)] for c in ("forward", "reversed", "none")],
                                              "fixed forward/reversed/none": [mean[(Fx, c)] for c in ("forward", "reversed", "none")],
                                              "contrast plastic/fixed": [contrast_p, contrast_f]}
            passes = passes and all(checks.values())
            kill_plastic_contrast = kill_plastic_contrast and contrast_p <= 0
            kill_vs_fixed = kill_vs_fixed and contrast_f >= 0.75 * contrast_p
    if kill_plastic_contrast or kill_vs_fixed:
        return "KILL", details
    return ("PASS" if passes else "PARTIAL"), details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4738)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/neuron_plasticity_smoke")
    p.add_argument("--stage1-iters", type=int, default=3000)
    p.add_argument("--stage2-iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--episodes-per-size", type=int, default=40)
    p.add_argument("--eval-circuits", type=int, default=12)
    args = p.parse_args()
    device = torch.device(args.device)
    out_dir = os.path.join(args.out_dir, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    data = make_train_data(args.seed, device, args.episodes_per_size)
    log(f"data ready {time.time() - t0:.0f}s")
    models = {}
    for name in ARMS:
        path = os.path.join(out_dir, f"{name}.pt")
        model = PlasticNeuronGraphDynamics(plastic=ARMS[name]).to(device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            log(f"{name} loaded")
        else:
            model = train_arm(name, data, args.seed, args, device, log)
            torch.save(model.state_dict(), path)
        models[name] = model
    out = evaluate(models, args.seed, args.eval_circuits, device, out_dir)
    verdict_name, details = verdict(out)
    summary = {"seed": args.seed, "args": vars(args), "parameters": {n: count_parameters(m) for n, m in models.items()},
               "verdict": verdict_name, "verdict_details": details, "episodes": out, "wall_seconds": time.time() - t0}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for key, d in details.items():
        log(f"{key}: plastic fwd/rev/none {[round(v[0], 2) for v in d['plastic forward/reversed/none']]} "
            f"teacher {[round(v[0], 2) for v in d['teacher forward/reversed/none']]} "
            f"fixed {[round(v[0], 2) for v in d['fixed forward/reversed/none']]} checks {d['checks']}")
    log(f"VERDICT {verdict_name}")


if __name__ == "__main__":
    main()
