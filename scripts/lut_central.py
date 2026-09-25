"""LUT for the CentralForceDynamics near field (cutoff 4, scripts/gravity_1b.py machinery): timing + profile at
N=1k..1M for the constant-density init (sparse, ~4 neighbours) and the same init compressed 4x (dense, ~64 neighbours),
then accuracy (one-step max diff, 200-step divergence / KE / momentum vs the MLP force).

Usage: PYTHONPATH=. python scripts/lut_central.py --out results/lut_central.json
"""
import argparse
import json
import time

import torch

from model.central_force import CentralForceDynamics
from model.token_graph import build_radius_graph_cells
from scripts.gravity_1b import init_state, step, tiled_accel
from scripts.lut_adaptive import central_force_fn

VARIANTS = ["base", "lut256", "lut1024", "lut4096", "lut1024c", "lut4096c"]


def fn_for(dyn, spec):
    if spec == "base":
        return central_force_fn(dyn)
    lut = int(spec[3:].rstrip("c"))
    return central_force_fn(dyn, lut, spec.endswith("c"))


def sync():
    torch.cuda.synchronize()


def make_state(count, regime, dev, seed):
    pos, vel, _ = init_state(count, dev, seed)
    if regime == "dense":
        pos = 500.0 + (pos - 500.0) * 0.25
    return pos, vel


@torch.no_grad()
def breakdown(force, pos, radius, reps=3):
    t = {"graph": 0.0, "force_mlp": 0.0, "gather_scatter": 0.0}
    for _ in range(reps):
        sync()
        a = time.perf_counter()
        edges = build_radius_graph_cells(pos, radius)
        sync()
        b = time.perf_counter()
        src, dst = edges[0], edges[1]
        rel = pos[src] - pos[dst]
        d = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
        sync()
        c = time.perf_counter()
        f = force(d)
        sync()
        e = time.perf_counter()
        torch.zeros_like(pos).index_add(0, dst, f * rel / d)
        sync()
        g = time.perf_counter()
        t["graph"] += b - a
        t["force_mlp"] += e - c
        t["gather_scatter"] += (c - b) + (g - e)
    return {k: 1000 * v / reps for k, v in t.items()}, int(edges.shape[1])


@torch.no_grad()
def time_step(force, pos, vel, dt, radius, strips, steps):
    p, v = step(force, pos, vel, dt, radius, strips)
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        p, v = step(force, p, v, dt, radius, strips)
    sync()
    return 1000 * (time.perf_counter() - t0) / steps


@torch.no_grad()
def run(force, pos, vel, dt, radius, steps, marks):
    out = {}
    for k in range(1, steps + 1):
        pos, vel = step(force, pos, vel, dt, radius, 1)
        if k in marks:
            out[k] = (pos.clone(), vel.clone())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--sizes", default="1000,10000,100000,1000000")
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/lut_central.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    dyn = CentralForceDynamics(dt=args.dt, neighbor_radius=args.radius).to(dev)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    forces = {s: fn_for(dyn, s) for s in VARIANTS}
    out = {"timing": [], "accuracy": {}, "table_error": {}}
    d = torch.rand(200000, 1, device=dev) * 4.0 + 1e-3
    with torch.no_grad():
        ref = forces["base"](d)
        for s in VARIANTS[1:]:
            e = (forces[s](d) - ref).abs()
            out["table_error"][s] = {"max_abs": float(e.max()), "ref_peak": float(ref.abs().max()), "max_rel_of_peak": float(e.max() / ref.abs().max())}
    print("table_error", json.dumps(out["table_error"]), flush=True)
    for regime in ("sparse", "dense"):
        for count in [int(s) for s in args.sizes.split(",")]:
            pos, vel = make_state(count, regime, dev, args.seed)
            strips = 1 if count <= 100000 else 4
            steps = 10 if count < 1_000_000 else 3
            row = {"regime": regime, "N": count, "variants": {}}
            row["breakdown_ms"], row["edges"] = breakdown(forces["base"], pos, args.radius)
            row["neighbours_per_body"] = row["edges"] / count
            for s in VARIANTS:
                row["variants"][s] = time_step(forces[s], pos, vel, args.dt, args.radius, strips, steps)
                print(regime, count, s, row["variants"][s], flush=True)
            print("ROW", json.dumps({k: v for k, v in row.items() if k != "variants"}), flush=True)
            out["timing"].append(row)
            json.dump(out, open(args.out, "w"))
    for regime in ("sparse", "dense"):
        for count in (1000, 10000):
            pos, vel = make_state(count, regime, dev, args.seed)
            marks = (1, 10, 50, 100, 200)
            base = run(forces["base"], pos, vel, args.dt, args.radius, 200, marks)
            rows = {}
            for s in VARIANTS[1:]:
                r = run(forces[s], pos, vel, args.dt, args.radius, 200, marks)
                rows[s] = {str(k): {"max_dp": float((r[k][0] - base[k][0]).norm(dim=1).max()),
                                    "mean_dp": float((r[k][0] - base[k][0]).norm(dim=1).mean()),
                                    "max_dv": float((r[k][1] - base[k][1]).norm(dim=1).max()),
                                    "KE_rel": float((0.5 * (r[k][1] ** 2).sum()) / (0.5 * (base[k][1] ** 2).sum()) - 1.0),
                                    "mom": float(r[k][1].double().sum(0).norm())} for k in marks}
            rows["base"] = {str(k): {"KE": float(0.5 * (base[k][1] ** 2).sum()), "mom": float(base[k][1].double().sum(0).norm())} for k in marks}
            out["accuracy"][f"{regime}_{count}"] = rows
            print("acc", regime, count, json.dumps(rows), flush=True)
            json.dump(out, open(args.out, "w"))
    print("done")


if __name__ == "__main__":
    main()
