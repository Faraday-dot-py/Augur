"""CURIOSITY experiment (no pre-registered prediction, no pass/kill verdict): finite-speed-local-field-medium.
A grid field with its own state, updated by a learned local rule, sourced by CIC-deposited mass; bodies read -grad(phi). Teacher-forced on exact softened-gravity
trajectories (G=1, eps 0.5, dt 0.1, 4 substeps). One model per propagation speed (n_sub substeps per tick; speed bound 2*h*n_sub/dt).
Observations written to results.json: force error vs exact instantaneous force at N=2, 20, 1000; cost per tick; per-tick error vs the retarded-force reference at N=2, 20
(raw and 3-tick filtered, share of ticks within 5%; reported, not scored).
Usage: PYTHONPATH=. python scripts/finite_speed_local_field_medium.py --device cuda"""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from model import local_field_medium as lfm
from scripts import scatter_scaling as ss
from scripts import scatter_scaling_ops as ops

EPS, DT, SUB = 0.5, 0.1, 4
EVAL_SEEDS = (9100, 9200, 9300)
STOP = False


def on_term(*_):
    global STOP
    STOP = True


def pair_scene(rng, extent):
    R = float(rng.uniform(4.0, 12.0))
    v = 0.5 * np.sqrt(2.0 / R)
    th = float(rng.uniform(0, 2 * np.pi))
    e = np.array([np.cos(th), np.sin(th)])
    t = np.array([-e[1], e[0]])
    return np.stack([-0.5 * R * e, 0.5 * R * e]), np.stack([-v * t, v * t])


def scene(n, seed, rep, extent):
    if n == 2:
        return pair_scene(np.random.default_rng([seed, 2, rep]), extent)
    return ss.make_scene(n, 6.0 if n <= 20 else 8.0, 0.3, extent, seed, rep)


def trajectory(n, seed, rep, ticks, extent, dev):
    p0, v0 = scene(n, seed, rep, extent)
    P, V = ops.exact_rollout(torch.tensor(p0, dtype=torch.float64, device=dev), torch.tensor(v0, dtype=torch.float64, device=dev), ticks, DT, SUB, EPS)
    A = torch.stack([ops.exact_acc(p, EPS) for p in P])
    return P.float(), A.float()


def retarded_acc(P, c):
    """P (T+1,N,2) positions per tick; force on i from j at the retarded time s solving s = t - |x_i(t)-x_j(s)|/(c*DT), history clamped to tick 0 (the medium starts empty)."""
    T1, N, _ = P.shape
    out = torch.zeros_like(P)
    ar = torch.arange(N, device=P.device)
    for t in range(T1):
        s = torch.full((N, N), float(t), device=P.device)
        for _ in range(8):
            f = s.floor().long().clamp(0, T1 - 2)
            w = (s - f.float()).clamp(0, 1)[..., None]
            src = P[f, ar[None, :]] * (1 - w) + P[f + 1, ar[None, :]] * w
            d = (P[t][:, None, :] - src).norm(dim=-1)
            s = (t - d / (c * DT)).clamp(min=0.0)
        rel = src - P[t][:, None, :]
        inv = ((rel ** 2).sum(-1) + EPS ** 2) ** -1.5
        inv[ar, ar] = 0.0
        out[t] = (rel * inv[..., None]).sum(1)
    return out


def rollout(model, P, nsub):
    B = 1
    N = P.shape[1]
    mass, mask = torch.ones(B, N, device=P.device), torch.ones(B, N, device=P.device)
    S = model.init_state(B, P.device)
    acc = []
    for t in range(P.shape[0]):
        S, a = model.tick(S, P[t][None], mass, mask, nsub)
        acc.append(a[0])
    return torch.stack(acc)


def rms_rel(a, ref):
    return float(((a - ref) ** 2).sum().sqrt() / ref.pow(2).sum().sqrt().clamp_min(1e-12))


def train_one(model, nsub, pools, args, dev, log):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + nsub)
    t0, it = time.time(), 0
    warm = args.train_warmup
    while it < args.iters and time.time() - t0 < args.train_seconds and not STOP:
        n = (2, 20)[it % 2]
        idx = rng.choice(len(pools[n]), args.batch, replace=False)
        P = torch.stack([pools[n][i][0] for i in idx], 1)
        A = torch.stack([pools[n][i][1] for i in idx], 1)
        mass, mask = torch.ones(args.batch, n, device=dev), torch.ones(args.batch, n, device=dev)
        S = model.init_state(args.batch, dev)
        loss = 0.0
        for t in range(P.shape[0]):
            S, a = model.tick(S, P[t], mass, mask, nsub)
            if t >= warm:
                loss = loss + ((a - A[t]) ** 2).sum() / A[t].pow(2).sum().clamp_min(1e-12)
        loss = loss / (P.shape[0] - warm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        it += 1
        if it % args.log_every == 0:
            log(f"nsub {nsub} it {it} loss {float(loss):.4f}")
    return it, time.time() - t0


@torch.no_grad()
def observe(model, nsub, args, dev):
    c = model.max_speed(nsub)
    cross = int(np.ceil(2 * args.extent / (c * DT)))
    skip = min(args.eval_ticks // 2, 2 * cross)
    out = {"speed_bound": c, "crossing_ticks": cross, "skipped_ticks": skip}
    for n in (2, 20, 1000):
        rows = []
        for seed in EVAL_SEEDS:
            P, A = trajectory(n, seed, 0, args.eval_ticks, args.extent, dev)
            a = rollout(model, P, nsub)
            row = {"seed": seed, "err_vs_exact": rms_rel(a[skip:], A[skip:])}
            if n <= 20:
                R = retarded_acc(P, c)
                filt = torch.stack([a[max(0, t - 2):t + 1].mean(0) for t in range(a.shape[0])])
                per = lambda x: torch.stack([((x[t] - R[t]) ** 2).sum().sqrt() / R[t].pow(2).sum().sqrt().clamp_min(1e-12) for t in range(skip, x.shape[0])])
                raw, flt = per(a), per(filt)
                row.update(err_vs_retarded=rms_rel(a[skip:], R[skip:]), retarded_vs_exact=rms_rel(R[skip:], A[skip:]),
                           within5_raw=float((raw <= 0.05).float().mean()), within5_filtered=float((flt <= 0.05).float().mean()),
                           median_per_tick_err_filtered=float(flt.median()))
            rows.append(row)
        out[f"n{n}"] = rows
    P, _ = trajectory(1000, EVAL_SEEDS[0], 0, 5, args.extent, dev)
    n_t = 20
    mass, mask = torch.ones(1, 1000, device=dev), torch.ones(1, 1000, device=dev)
    S = model.init_state(1, dev)
    for _ in range(3):
        S, _ = model.tick(S, P[0][None], mass, mask, nsub)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_t):
        S, _ = model.tick(S, P[0][None], mass, mask, nsub)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    out["ms_per_tick_n1000"] = (time.time() - t0) / n_t * 1e3
    out["field_substeps_per_tick"] = nsub
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsubs", default="1,2,4,8")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--extent", type=float, default=64.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-ticks", type=int, default=32)
    ap.add_argument("--train-warmup", type=int, default=8)
    ap.add_argument("--train-scenes", type=int, default=64)
    ap.add_argument("--eval-ticks", type=int, default=64)
    ap.add_argument("--train-seconds", type=float, default=600.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/finite_speed_local_field_medium")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, on_term)
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    res_path = f"{args.out}/results.json"
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res.setdefault("label", "curiosity")
    res["note"] = "curiosity: observations only, no pass/kill verdict"
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    log(f"curiosity run, args {vars(args)}")
    torch.manual_seed(args.seed)
    pools = {n: [trajectory(n, args.seed, r, args.train_ticks - 1, args.extent, dev) for r in range(args.train_scenes)] for n in (2, 20)}
    log(f"train pools {[(n, len(v), tuple(v[0][0].shape)) for n, v in pools.items()]}")
    for nsub in (int(x) for x in args.nsubs.split(",")):
        if STOP:
            raise SystemExit(143)
        key = f"nsub{nsub}"
        if key in res and "observations" in res[key]:
            log(f"{key} done, skip")
            continue
        torch.manual_seed(args.seed + nsub)
        model = lfm.LocalFieldMedium(args.grid, args.extent, args.width, DT).to(dev)
        its, secs = train_one(model, nsub, pools, args, dev, log)
        if STOP:
            raise SystemExit(143)
        model.eval()
        obs = observe(model, nsub, args, dev)
        res[key] = {"train_iters": its, "train_seconds": secs, "observations": obs}
        json.dump(res, open(res_path, "w"), indent=1)
        torch.save(model.state_dict(), f"{args.out}/{key}.pt")
        log(f"{key} saved: n2 err {np.mean([r['err_vs_exact'] for r in obs['n2']]):.3f} n20 err {np.mean([r['err_vs_exact'] for r in obs['n20']]):.3f} "
            f"n1000 err {np.mean([r['err_vs_exact'] for r in obs['n1000']]):.3f} ms/tick {obs['ms_per_tick_n1000']:.2f}")
    done = [k for k in res if k.startswith("nsub")]
    if len(done) != len(args.nsubs.split(",")):
        raise SystemExit(f"expected {len(args.nsubs.split(','))} speeds, have {len(done)}")
    log(f"complete: {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
