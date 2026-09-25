"""Timing of LUT / adaptive-substep variants of the conservative-contact model (GPU), sparse and dense regimes,
plus a per-section profile of the unmodified step. See scripts/lut_adaptive.py for variant specs.

Usage: PYTHONPATH=. python scripts/lut_bench.py --sizes 1000,10000,100000,1000000 --out results/lut_bench.json
"""
import argparse
import json
import math
import random
import time

import torch

from model.token_graph import build_radius_graph_cells
from scripts.energy_fix_common import build_model
from scripts.lut_adaptive import install

R, DT = 0.75, 0.15


def sync():
    torch.cuda.synchronize()


def sparse_state(count, device, seed):
    n = max(100, math.ceil(10 * math.sqrt(count)))
    gen = torch.Generator().manual_seed(seed)
    pos = (torch.rand(count, 2, generator=gen) * (n - 3) + 1.5).to(device)
    vel = ((torch.rand(count, 2, generator=gen) - 0.5) * 4.6).to(device)
    return n, pos, vel


def dense_state(count, device, seed, rows_max=30, spacing=1.45):
    n = max(100, math.ceil(spacing * count / rows_max) + 3)
    cols = int((n - 3) / spacing)
    k = torch.arange(count)
    row, col = k // cols, k % cols
    gen = torch.Generator().manual_seed(seed)
    pos = torch.stack([(n - 1) - R - spacing * row.double(), 1.5 + spacing * col.double()], dim=1).float()
    pos = pos + 0.02 * torch.randn(count, 2, generator=gen)
    return n, pos.to(device), torch.zeros(count, 2, device=device)


def contact_fraction(pos, thresh=2 * R):
    e = build_radius_graph_cells(pos, thresh)
    m = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    m[e[0]] = True
    return float(m.float().mean()), int(e.shape[1])


class Timer:
    def __init__(self):
        self.t = {}
        self.last = None

    def start(self):
        sync()
        self.last = time.perf_counter()

    def mark(self, name):
        sync()
        now = time.perf_counter()
        self.t[name] = self.t.get(name, 0.0) + now - self.last
        self.last = now


def breakdown(dyn, pos, vel, reps):
    T = Timer()
    n, r = dyn.n, dyn.radius
    h = dyn.dt / dyn.contact_substeps

    def accel(p):
        x, y = p[:, 0], p[:, 1]
        d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
        pen = (r - d).clamp(min=0.0) / r
        f = pen * dyn.wall_force(pen.unsqueeze(-1)).squeeze(-1) * dyn.force_scale
        acc = torch.stack([f[:, 0] - f[:, 1], f[:, 2] - f[:, 3]], dim=1) + dyn.gravity * 10.0
        T.mark("wall")
        edges = build_radius_graph_cells(p, 2 * r)
        T.mark("graph")
        if edges.shape[1] > 0:
            src, dst = edges[0], edges[1]
            rel = p[dst] - p[src]
            dist = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
            pen = (2 * r - dist).clamp(min=0.0) / (2 * r)
            T.mark("pair_geom")
            fp = pen * dyn.pair_force(pen) * dyn.force_scale
            T.mark("pair_mlp")
            acc = acc.index_add(0, dst, fp * rel / dist)
            T.mark("pair_scatter")
        return acc

    with torch.no_grad():
        for _ in range(reps):
            p, v = pos, vel
            T.start()
            a = accel(p)
            for _ in range(dyn.contact_substeps):
                vh = v + 0.5 * h * a
                p = p + h * vh
                T.mark("update")
                a = accel(p)
                v = vh + 0.5 * h * a
                T.mark("update")
    return {k: 1000 * s / reps for k, s in T.t.items()}


def kernel_count(model, pos, vel, hid):
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        model.step_free(pos, vel, hid, render=False)
        sync()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            model.step_free(pos, vel, hid, render=False)
            sync()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return {"kernels": len(ev), "gpu_kernel_ms": sum(getattr(e, 'device_time', None) or getattr(e, 'cuda_time', 0) for e in ev) / 1000.0}


def time_variant(model, spec, pos0, vel0, steps, warm=2):
    fc = install(model, spec)
    pos, vel = pos0.clone(), vel0.clone()
    hid = torch.zeros(pos.shape[0], model.dynamics.hidden_dim, device=pos.device)
    with torch.no_grad():
        for _ in range(warm):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
        sync()
        if fc is not None:
            fc.stats = {"active": [], "n": []}
        t0 = time.perf_counter()
        for _ in range(steps):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
        sync()
        ms = 1000 * (time.perf_counter() - t0) / steps
    out = {"ms_per_step": ms, "finite": bool(torch.isfinite(pos).all())}
    if fc is not None and fc.stats["active"]:
        out["active_frac"] = sum(fc.stats["active"]) / sum(fc.stats["n"])
    install(model, "base")
    return out, pos, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cons_pure.pt")
    ap.add_argument("--sizes", default="1000,10000,100000,1000000")
    ap.add_argument("--regimes", default="sparse,dense")
    ap.add_argument("--variants", default="base,lut256,lut1024,lut4096,lut1024c,lut4096c,skin0,skin1,skin2,lut1024+skin1,lut4096c+skin1")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/lut_bench.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows = []
    for regime in args.regimes.split(","):
        for count in [int(s) for s in args.sizes.split(",")]:
            n, pos, vel = (sparse_state if regime == "sparse" else dense_state)(count, dev, args.seed)
            model = build_model("cons", args.checkpoint, n, dev)
            model.dynamics.cell_graph = True
            hid = torch.zeros(count, model.dynamics.hidden_dim, device=dev)
            warm = 5 if regime == "sparse" else 20
            with torch.no_grad():
                for _ in range(warm):
                    pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            frac, ne = contact_fraction(pos)
            steps = args.steps if count < 1_000_000 else max(3, args.steps // 3)
            row = {"regime": regime, "N": count, "grid_n": n, "contact_frac": frac, "edges": ne,
                   "vmax": float(vel.norm(dim=1).max()), "variants": {}}
            row["breakdown_ms"] = breakdown(model.dynamics, pos, vel, 2 if count >= 1_000_000 else 4)
            if count <= 10000:
                row["kernels"] = kernel_count(model, pos, vel, hid)
            for spec in args.variants.split(","):
                res, _, _ = time_variant(model, spec, pos, vel, steps)
                row["variants"][spec] = res
                print(regime, count, spec, json.dumps(res), flush=True)
            print("ROW", json.dumps({k: v for k, v in row.items() if k != "variants"}), flush=True)
            rows.append(row)
            json.dump(rows, open(args.out, "w"))
            del model, pos, vel, hid
            torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
