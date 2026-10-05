"""Speed/memory benchmark for the scatter-field model (inference ms/tick vs N, training it/s, torch.profiler top ops).

Usage: PYTHONPATH=. python scripts/scatter_opt_bench.py --mode infer|train|profile [--ns 2,10,100,300,1000] [--tag name] [--opt flag,flag]
Weights: Exp E checkpoint if --ckpt given, else random init (timing only).
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf


def sync():
    torch.cuda.synchronize()


def make_model(args, dev):
    m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    if args.ckpt:
        m.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    else:
        torch.manual_seed(4738)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
    for o in [x for x in args.opt.split(",") if x]:
        sf.apply_opt(m, o)
    return m


def scene(n, B, dev, seed=4738, cold=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = max(8.0, 0.5 * n ** 0.5)
    s = min(s, 28.0)
    if cold:
        s = 16.0
    pos = (torch.randn(B, n, 2, generator=g) * s / 2).clamp(-29, 29).to(dev)
    vel = (torch.randn(B, n, 2, generator=g) * 0.3).to(dev)
    return pos, vel, torch.ones(B, n, device=dev), torch.ones(B, n, device=dev)


def infer(args, dev):
    m = make_model(args, dev)
    out = {}
    for n in [int(x) for x in args.ns.split(",")]:
        B = args.batch
        pos, vel, mass, mask = scene(n, B, dev, cold=args.cold)
        steps = args.steps if n <= 1000 else max(5, args.steps // 10)
        try:
            sf.rollout(m, pos, vel, mass, mask, 3)
            sync()
            base = torch.cuda.memory_allocated(dev)
            ts = []
            for _ in range(3):
                torch.cuda.reset_peak_memory_stats(dev)
                sync()
                t = time.perf_counter()
                P, V = sf.rollout(m, pos, vel, mass, mask, steps)
                sync()
                ts.append((time.perf_counter() - t) / steps * 1000)
            peak = (torch.cuda.max_memory_allocated(dev) - base) / 2 ** 20
            out[n] = {"ms": float(np.median(ts)), "peak_mb": peak, "final_pos_sum": float(P[:, -1].double().sum()), "finite": bool(torch.isfinite(P).all())}
            torch.save({"P": P.cpu(), "V": V.cpu()}, f"results/{args.tag}_infer_n{n}.pt") if args.save else None
        except torch.cuda.OutOfMemoryError:
            out[n] = {"ms": None, "peak_mb": None, "error": "OOM"}
            torch.cuda.empty_cache()
        print(f"N {n} B {B}: {out[n]}", flush=True)
    json.dump(out, open(f"results/{args.tag}_infer.json", "w"), indent=1)


def train(args, dev):
    m = make_model(args, dev)
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    B, n, k = args.batch, args.n_train, args.k
    pos0, vel0, mass, mask = scene(n, B, dev)
    pos0 = pos0 * 0.5
    tgt = pos0 + 0.1
    fns = sf.make_step_fns(m, B, n, k, dev) if "graphtrain" in args.opt else None

    def it():
        pos, vel, f = pos0, vel0, m.init_field(B, dev)
        loss = 0.0
        for s in range(k):
            pos, vel, _, f, _ = (fns[s] if fns else lambda *a: sf._step_m(m, *a))(pos, vel, mass, mask, f)
            loss = loss + ((pos - tgt) ** 2).mean() + 0.1 * (vel ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        return loss

    for _ in range(3):
        it()
    sync()
    torch.cuda.reset_peak_memory_stats(dev)
    t = time.perf_counter()
    for _ in range(args.iters):
        l = it()
    sync()
    dt = (time.perf_counter() - t) / args.iters
    res = {"it_per_s": 1 / dt, "ms_per_it": dt * 1000, "peak_mb": torch.cuda.max_memory_allocated(dev) / 2 ** 20, "loss": float(l)}
    print(f"train B {B} N {n} k {k}: {res}", flush=True)
    json.dump(res, open(f"results/{args.tag}_train.json", "w"), indent=1)


def profile_train(args, dev):
    from torch.profiler import ProfilerActivity, profile as prof

    m = make_model(args, dev)
    B, n, k = args.batch, args.n_train, args.k
    pos0, vel0, mass, mask = scene(n, B, dev)
    pos0 = pos0 * 0.5

    def it():
        pos, vel, f = pos0, vel0, m.init_field(B, dev)
        loss = 0.0
        for s in range(k):
            pos, vel, _, f, _ = sf._step_m(m, pos, vel, mass, mask, f)
            loss = loss + (pos ** 2).mean() + 0.1 * (vel ** 2).mean()
        m.zero_grad()
        loss.backward()

    for _ in range(2):
        it()
    sync()
    t = time.perf_counter()
    with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        it()
        sync()
    print(f"wall {(time.perf_counter() - t) * 1000:.1f} ms (with profiler)")
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))


def profile(args, dev):
    from torch.profiler import ProfilerActivity, profile as prof

    m = make_model(args, dev)
    for n in [int(x) for x in args.ns.split(",")]:
        pos, vel, mass, mask = scene(n, args.batch, dev)
        sf.rollout(m, pos, vel, mass, mask, 3)
        sync()
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            sf.rollout(m, pos, vel, mass, mask, 10)
            sync()
        print(f"=== profile N {n} B {args.batch}, 10 ticks (cuda time)", flush=True)
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60), flush=True)
        print(p.key_averages().table(sort_by="cpu_time_total", row_limit=8, max_name_column_width=60), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--ns", default="2,10,100,300,1000")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opt", default="")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    {"infer": infer, "train": train, "profile": profile, "profile_train": profile_train}[args.mode](args, dev)


if __name__ == "__main__":
    main()
