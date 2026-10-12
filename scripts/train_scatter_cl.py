"""Staged closed-loop relabel curriculum for the scatter-field model (Exp E line), and its same-schedule truth-window control.

Stage s in OFFSETS: a training window starts from the model's own state at tick s (current model rolled s ticks from a truth initial state, no grad),
and the target is the exact fp64 sim (gravity_sim scheme: KDK, 4 substeps/tick, softened G=1) run k ticks from that state. Stage 0 is the ordinary
truth-window batch of scripts/train_scatter_field.py. Later stages mix --mix-truth stage-0 batches. --mode truth is the control: every batch is a
truth window (same loop, lr schedule, k curriculum, loss, iterations).

Relabel batches are regenerated every iteration from the current model (no cache: the roll-in drifts with the weights, a cache would relabel stale states).
Gate: advance when the relabelled held-out err@20 at the current offset (eval seeds 9000/12000, mean) improves < --plateau for --patience consecutive evals
after >= --min-stage-its iterations in the stage, or after --max-stage-its; the truth-IC err@20 is logged next to it.

Usage: PYTHONPATH=. python scripts/train_scatter_cl.py --mode relabel --init-ckpt <E.pt> --name <n> --iters 12000 --device cuda
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

OFFSETS = (0, 5, 10, 20)


def accel(p, mass, mask, eps=0.5, g=1.0):
    """Exact softened all-pairs acceleration, p (B,N,2), mass, mask (B,N)."""
    d = p[:, None, :, :] - p[:, :, None, :]
    eye = 1.0 - torch.eye(p.shape[1], device=p.device, dtype=p.dtype)
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5 * eye
    return g * (d * inv[..., None] * (mass * mask)[:, None, :, None]).sum(2) * mask[..., None]


def exact_rollout(pos, vel, mass, mask, steps, dt=0.1, substeps=4, eps=0.5, g=1.0):
    """Batched exact softened sim, same KDK scheme as gravity_sim.rollout_torch. pos, vel (B,N,2), mass, mask (B,N), all fp64. Returns (B,steps+1,N,2) x2."""
    h = dt / substeps

    def acc(p):
        return accel(p, mass, mask, eps, g)

    ps, vs = [pos], [vel]
    a = acc(pos)
    for _ in range(steps):
        for _ in range(substeps):
            vel = vel + 0.5 * h * a
            pos = pos + h * vel
            a = acc(pos)
            vel = vel + 0.5 * h * a
        ps.append(pos)
        vs.append(vel)
    return torch.stack(ps, 1), torch.stack(vs, 1)


def roll_in(model, pos, vel, m, mk, s):
    """Current model rolled s ticks, no grad. Returns detached (pos, vel, m, recurrent state)."""
    state = model.init_field(pos.shape[0], pos.device)
    with torch.no_grad():
        for _ in range(s):
            pos, vel, m, state, _ = sf._step_m(model, pos, vel, m, mk, state)
    return pos, vel, m, tsf.detach_state(state)


def perturb(pos, vel, mk, sigma, g):
    """Zero-mean (per scene) gaussian noise on pos and vel, masked bodies untouched."""
    w = mk[..., None]
    out = []
    for x in (pos, vel):
        n = torch.randn(x.shape, generator=g).to(x.device) * w
        n = n - (n.sum(1, keepdim=True) / mk.sum(1)[:, None, None]) * w
        out.append(x + sigma * n)
    return out


def relabel(pos, vel, m, mk, k, dt=0.1, eps=0.5):
    """Exact fp64 sim from the (model-generated) state; returns float32 targets (B,k+1,N,2) x2."""
    Pt, Vt = exact_rollout(pos.double(), vel.double(), m.double(), mk.double(), k, dt=dt, eps=eps)
    return Pt.float(), Vt.float()


def relabel_batch(model, P0, V0, m, mk, s, k, args, g=None):
    """Roll in s ticks from the truth IC (P0, V0), optionally perturb, relabel k ticks. Returns pos, vel, state, Pt, Vt."""
    pos, vel, m, state = roll_in(model, P0, V0, m, mk, s)
    if args.noise > 0 and g is not None:
        pos, vel = perturb(pos, vel, mk, args.noise, g)
        state = (state[0], None)
    if args.reset_field:
        state = model.init_field(pos.shape[0], pos.device)
    Pt, Vt = relabel(pos, vel, m, mk, k, args.dt, args.eps)
    return pos, vel, state, Pt, Vt


@torch.no_grad()
def eval_err(model, ev, s, k, args):
    """Held-out err@k (mean per-body position error, sim units). s = 0: standard truth-IC rollout vs the stored truth; s > 0: relabelled windows."""
    P, V, M, mask = ev
    Mf, mk = M.float(), mask.float()
    if s == 0:
        pos, vel, state = P[:, 0].float(), V[:, 0].float(), model.init_field(P.shape[0], P.device)
        Pt = P[:, :k + 1]
    else:
        pos, vel, _, state = roll_in(model, P[:, 0].float(), V[:, 0].float(), Mf, mk, s)
        Pt = relabel(pos, vel, Mf, mk, k, args.dt, args.eps)[0].double()
    for _ in range(k):
        pos, vel, _, state, _ = sf._step_m(model, pos, vel, Mf, mk, state)
    e = ((pos.double() - Pt[:, k]) ** 2).sum(-1).sqrt() * mask
    return float((e.sum(1) / mask.sum(1)).mean())


def evaluate(model, evs, args):
    """{offset: mean over eval seeds of err@20}; offset 0 is the truth-IC eval."""
    return {s: float(np.mean([eval_err(model, ev, s, 20, args) for ev in evs])) for s in OFFSETS}


def train(model, tensors, evs, args, ckpt_path, log):
    P, V, M, mask = [t.float() for t in tensors]
    S = P.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, stage, stage_start, best, bad, hist, decisions = 0, 0, 0, None, 0, [], []
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, stage, stage_start, best, bad, hist, decisions = st["it"], st["stage"], st["stage_start"], st["best"], st["bad"], st["hist"], st["decisions"]
        log(f"resumed from it {start} stage {stage}")
    g = torch.Generator(device="cpu").manual_seed(args.seed + start)
    fns = sf.make_step_fns(model, args.batch, P.shape[2], args.k_end, P.device) if "graphtrain" in getattr(model, "opt", ()) else None

    def save(it):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "stage": stage, "stage_start": stage_start, "best": best, "bad": bad,
                    "hist": hist, "decisions": decisions}, ckpt_path)

    skipped = nrel = 0
    t_start = time.time()
    it = start - 1
    while True:
        it += 1
        if it >= args.iters:
            break
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        if evs and it % args.eval_every == 0 and not (start > 0 and it == start):
            ev = evaluate(model, evs, args)
            s = OFFSETS[stage]
            hist.append({"it": it, "stage": stage, "err20": {str(a): b for a, b in ev.items()}})
            log(f"eval it {it} stage {stage} (offset {s}) err@20 by offset {({a: round(b, 5) for a, b in ev.items()})}")
            if args.mode == "relabel" and stage < len(OFFSETS) - 1:
                cur = ev[s]
                if best is None or cur < best * (1 - args.plateau):
                    bad = 0
                else:
                    bad += 1
                best = cur if best is None else min(best, cur)
                n = it - stage_start
                why = "plateau" if (bad >= args.patience and n >= args.min_stage_its) else "cap" if n >= args.max_stage_its else None
                if why:
                    decisions.append({"it": it, "from": s, "to": OFFSETS[stage + 1], "why": why, "relabel_err20": cur, "best": best, "truth_ic_err20": ev[0]})
                    log(f"GATE it {it}: advance offset {s} -> {OFFSETS[stage + 1]} ({why}) relabel err@20 {cur:.5f} best {best:.5f} truth-IC err@20 {ev[0]:.5f}")
                    stage, stage_start, best, bad = stage + 1, it, None, 0
            save(it)
        s = OFFSETS[stage] if args.mode == "relabel" else 0
        idx = torch.randint(S, (args.batch,), generator=g).to(args.device)
        use_truth = s == 0 or torch.rand(1, generator=g).item() < args.mix_truth
        m, mk = M[idx], mask[idx]
        if use_truth:
            t0 = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(args.device)
            tt = t0[:, None] + torch.arange(k + 1, device=args.device)[None]
            Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
            pos, vel = Pw[:, 0], Vw[:, 0]
            field = model.init_field(args.batch, args.device)
        else:
            pos, vel, field, Pw, Vw = relabel_batch(model, P[idx, 0], V[idx, 0], m, mk, s, k, args, g)
            nrel += 1
        lp = lv = lm = 0.0
        w = mk[:, :, None]
        denom = mk.sum() * 2 * k
        m0 = m
        for i in range(k):
            pos, vel, m, field, _ = fns[i](pos, vel, m, mk, field) if fns else sf._step_m(model, pos, vel, m, mk, field)
            lm = lm + (((m - m0) ** 2) * mk).sum() / (mk.sum() * k)
            lp = lp + (((pos - Pw[:, i + 1]) ** 2) * w).sum() / denom
            lv = lv + (((vel - Vw[:, i + 1]) ** 2) * w).sum() / denom
        loss = lp + 0.1 * lv + lm
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn) and (args.loss_cap <= 0 or loss.item() < args.loss_cap):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} non-finite/capped loss {loss.item():.4g}, step skipped ({skipped} total)")
        if it % args.log_every == 0:
            log(f"it {it} k {k} stage {stage} {'truth' if use_truth else 'relabel'} loss {loss.item():.6f} t {time.time() - t_start:.0f}s")
    if all(torch.isfinite(p).all() for p in model.parameters()):
        save(it)
    log(f"trained to it {it} in {time.time() - t_start:.0f}s, {skipped} skipped, {nrel} relabel batches, decisions {decisions}")
    return it, hist, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["relabel", "truth"], default="relabel")
    ap.add_argument("--name", required=True)
    ap.add_argument("--exp", default="E")
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--opt", default="default")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--eval-scenes", type=int, default=48)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loss-cap", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--mix-truth", type=float, default=0.25)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--reset-field", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--min-stage-its", type=int, default=1000)
    ap.add_argument("--max-stage-its", type=int, default=3000)
    ap.add_argument("--plateau", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out-dir", default="results/scatter_cl")
    ap.add_argument("--ckpt-dir", default="checkpoints/scatter_cl")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    dd = dev if dev.type == "cuda" else None
    cfg = tsf.EXPS[args.exp]

    def log(s):
        print(s, flush=True)

    model = tsf.build(cfg, args.variant, args.dt).to(dev)
    model.load_state_dict(torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
    sf.apply_opts(model, args.opt, "train")
    tensors = sf.to_tensors(tsf.make_data(cfg, args.train, args.steps, args.seed, dd, args.dt, args.eps), dev)
    eval_data = {seed: tsf.make_data(cfg, args.eval_scenes, args.eval_steps, seed, dd, args.dt, args.eps) for seed in tsf.EVAL_SEEDS}
    evs = [sf.to_tensors(eval_data[seed], dev) for seed in tsf.EVAL_SEEDS]
    ckpt = f"{args.ckpt_dir}/{args.name}.pt"
    log(f"=== {args.mode} {args.name} exp {args.exp} {args.variant} iters {args.iters} lr {args.lr} noise {args.noise} mix {args.mix_truth} reset_field {args.reset_field}")
    log(f"init eval err@20 by offset {evaluate(model, evs, args)}")
    n_it, hist, decisions = train(model, tensors, evs, args, ckpt, log)
    final = evaluate(model, evs, args)
    log(f"final eval err@20 by offset {final}")
    json.dump({"args": vars(args), "iters": n_it, "hist": hist, "decisions": decisions, "final_err20": {str(a): b for a, b in final.items()}},
              open(f"{args.out_dir}/{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
