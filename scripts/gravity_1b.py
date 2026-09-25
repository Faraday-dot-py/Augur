"""Strip-tiled free rollout of CentralForceDynamics for very large N (GPU). Bodies
stay resident as (N, 2) pos/vel; each acceleration evaluation runs one x-strip at
a time together with the halo of bodies within `neighbor_radius` of its edges, so
no body's neighbour set depends on the tiling. Force is truncated at
`neighbor_radius` (an all-pairs sum is infeasible at this N). Initial state is the
constant-density scaled init from gravity_sim.init_bodies, drawn on the device.

Usage: PYTHONPATH=. python3 scripts/gravity_1b.py --bodies 1000000000 --out results/gravity_1b.npz
"""
import argparse
import json
import time

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from model.token_graph import build_radius_graph_cells


def init_state(n, device, seed):
    scale = n / 8
    spread, speed = 5.0 * scale ** 0.5, 0.5 * scale ** 0.25
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pos = torch.rand(n, 2, generator=gen, device=device) * (2 * spread) + (500.0 - spread)
    vel = torch.randn(n, 2, generator=gen, device=device) * speed
    vel -= vel.mean(0)
    return pos, vel, spread


def model_force(dyn):
    return lambda d: dyn.force(torch.log(d)) / (d ** 2 + 1.0)


def analytic_force(eps):
    return lambda d: d / (d ** 2 + eps ** 2) ** 1.5


@torch.no_grad()
def tiled_accel(force_fn, pos, radius, strips):
    x = pos[:, 0].contiguous()
    lo, hi = float(x.min()), float(x.max()) + 1e-3
    bounds = np.linspace(lo, hi, strips + 1)
    acc = torch.zeros_like(pos)
    for s in range(strips):
        a, b = float(bounds[s]), float(bounds[s + 1])
        idx = ((x >= a - radius) & (x < b + radius)).nonzero().squeeze(1)
        p = pos[idx]
        is_core = (p[:, 0] >= a) & (p[:, 0] < b)
        edges = build_radius_graph_cells(p, radius)
        keep = is_core[edges[1]]
        src, dst = edges[0][keep], edges[1][keep]
        rel = p[src] - p[dst]
        d = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
        chunk = torch.zeros_like(p).index_add(0, dst, force_fn(d) * rel / d)
        acc[idx[is_core]] = chunk[is_core]
    return acc


@torch.no_grad()
def step(force_fn, pos, vel, dt, radius, strips):
    a0 = tiled_accel(force_fn, pos, radius, strips)
    new_pos = pos + vel * dt + 0.5 * dt * dt * a0
    a1 = tiled_accel(force_fn, new_pos, radius, strips)
    vel = vel + 0.5 * dt * (a0 + a1)
    return new_pos, vel


@torch.no_grad()
def density_maps(pos, vel, spread, grid, chunk=100_000_000):
    half = spread * 1.25
    cell = 2 * half / grid
    dens = torch.zeros(grid * grid, device=pos.device)
    spd = torch.zeros(grid * grid, device=pos.device)
    for i in range(0, pos.shape[0], chunk):
        p, v = pos[i:i + chunk], vel[i:i + chunk]
        ij = ((p - (500.0 - half)) / cell).floor().long()
        ok = (ij >= 0).all(1) & (ij < grid).all(1)
        flat = ij[ok, 0] * grid + ij[ok, 1]
        dens += torch.bincount(flat, minlength=grid * grid).float()
        spd += torch.bincount(flat, weights=v[ok].norm(dim=1), minlength=grid * grid).float()
    return dens.view(grid, grid).cpu().numpy(), (spd / dens.clamp(min=1)).view(grid, grid).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--bodies", type=int, default=1_000_000_000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--strips", type=int, default=100)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--compare-analytic", action="store_true", help="also run the analytic truncated force from the same init and report the position gap")
    ap.add_argument("--out", default="results/gravity_1b.npz")
    args = ap.parse_args()

    dev = torch.device(args.device)
    dyn = CentralForceDynamics(dt=args.dt, neighbor_radius=args.radius).to(dev)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    force = model_force(dyn)
    pos, vel, spread = init_state(args.bodies, dev, args.seed)
    ref = None
    if args.compare_analytic:
        ref = (pos.clone(), vel.clone())
    dens, spd, steps, ke, mom, gap = [], [], [], [], [], []
    t0 = time.time()

    def record(k):
        d, s = density_maps(pos, vel, spread, args.grid)
        dens.append(d)
        spd.append(s)
        steps.append(k)
        ke.append(float(0.5 * (vel.double() ** 2).sum()))
        mom.append(float(vel.double().sum(0).norm()))
        if ref is not None:
            gap.append(float((pos - ref[0]).norm(dim=1).mean()))
        np.savez_compressed(args.out, density=np.stack(dens), speed=np.stack(spd), steps=steps, ke=ke, momentum=mom,
                            gap=gap, spread=spread, bodies=args.bodies, radius=args.radius, dt=args.dt)

    record(0)
    for k in range(1, args.steps + 1):
        ts = time.time()
        pos, vel = step(force, pos, vel, args.dt, args.radius, args.strips)
        if ref is not None:
            ref = step(analytic_force(args.eps), ref[0], ref[1], args.dt, args.radius, args.strips)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        print(f"step {k} {time.time() - ts:.1f}s elapsed {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0:.0f}GB", flush=True)
        if k % args.record_every == 0:
            record(k)
            print(f"  KE {ke[-1]:.4g} |P| {mom[-1]:.4g}" + (f" gap {gap[-1]:.4f}" if gap else ""), flush=True)
        if time.time() - t0 > args.max_seconds:
            print("time budget reached", flush=True)
            break
    print(json.dumps({"steps_done": k, "wall_s": time.time() - t0}))


if __name__ == "__main__":
    main()
