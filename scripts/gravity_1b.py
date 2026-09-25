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


FORCE_TRAIN_MAX = 150.0


def init_state(n, device, seed):
    scale = n / 8
    spread, speed = 5.0 * scale ** 0.5, 0.5 * scale ** 0.25
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pos = torch.rand(n, 2, generator=gen, device=device) * (2 * spread) + (500.0 - spread)
    vel = torch.randn(n, 2, generator=gen, device=device) * speed
    vel -= vel.mean(0)
    return pos, vel, spread


def init_clusters(clusters, per_cluster, spacing, device, seed):
    """`clusters` independent constant-density scaled-init clusters of
    `per_cluster` bodies each (zero net momentum per cluster) on a square
    lattice of pitch `spacing`, cluster-major body order. Returns the lattice
    half-width as the view half-width."""
    scale = per_cluster / 8
    spread, speed = 5.0 * scale ** 0.5, 0.5 * scale ** 0.25
    side = int(np.ceil(clusters ** 0.5))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k = torch.arange(clusters, device=device)
    centres = torch.stack([k // side, k % side], dim=1).float() * spacing + (500.0 - (side - 1) * spacing / 2)
    pos = (torch.rand(clusters, per_cluster, 2, generator=gen, device=device) * (2 * spread) - spread) + centres[:, None, :]
    vel = torch.randn(clusters, per_cluster, 2, generator=gen, device=device) * speed
    vel -= vel.mean(1, keepdim=True)
    return pos.reshape(-1, 2), vel.reshape(-1, 2), side * spacing / 2 * 1.05 / 1.25


def model_force(dyn):
    return lambda d: dyn.force(torch.log(d.clamp(max=FORCE_TRAIN_MAX))) / (d ** 2 + 1.0)


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
def far_kernel(force_fn, h, radius, grid, dev, sub=6, reach=None, fft64=False):
    """Mesh kernel for the far field, indexed by dst-src cell offset on a
    (2*grid, 2*grid) periodic layout. Offsets within `reach` cells average the
    force over sub x sub points per cell with d <= radius masked out (the
    near field covers those pairs exactly); farther offsets use the cell-centre
    force."""
    fn = force_fn
    force_fn = lambda d: torch.cat([fn(c.reshape(-1, 1)) for c in d.reshape(-1).split(8_000_000)]).reshape(d.shape)
    reach = reach if reach is not None else int(np.ceil(radius / h)) + 2
    n = 2 * grid
    o = torch.arange(n, device=dev)
    o = torch.where(o >= grid, o - n, o).float()
    ox, oy = torch.meshgrid(o, o, indexing="ij")
    rx, ry = -ox * h, -oy * h
    d = torch.sqrt(rx ** 2 + ry ** 2 + 1e-12)
    f = force_fn(d) / d
    f = torch.where(d > radius, f, torch.zeros_like(f))
    kx, ky = f * rx, f * ry
    u = (torch.arange(sub, device=dev).float() + 0.5) / sub * h
    du = (u[:, None] - u[None, :]).reshape(-1)
    ddx, ddy = torch.meshgrid(du, du, indexing="ij")
    ddx, ddy = ddx.reshape(-1), ddy.reshape(-1)
    r = torch.arange(-reach, reach + 1, device=dev).float()
    for i, a in enumerate(r):
        for j, b in enumerate(r):
            x = -a * h + ddx
            y = -b * h + ddy
            dd = torch.sqrt(x ** 2 + y ** 2 + 1e-12)
            ff = torch.where(dd > radius, force_fn(dd) / dd, torch.zeros_like(dd))
            ix, iy = int(a) % n, int(b) % n
            kx[ix, iy] = (ff * x).mean()
            ky[ix, iy] = (ff * y).mean()
    if fft64:
        kx, ky = kx.double(), ky.double()
    return torch.fft.rfft2(kx), torch.fft.rfft2(ky)


def cic_corners(g, grid):
    i0 = g.floor().long()
    w = g - i0
    for dx in (0, 1):
        for dy in (0, 1):
            ix = (i0[:, 0] + dx).clamp(0, grid - 1)
            iy = (i0[:, 1] + dy).clamp(0, grid - 1)
            wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1])
            yield ix * grid + iy, wt


@torch.no_grad()
def far_accel(force_fn, pos, radius, grid, chunk=100_000_000, mode="cic", fft64=True):
    """Particle-mesh far field: deposit counts on a grid over the current
    bounding box, convolve with the force kernel by FFT, interpolate back.
    mode "cic" (default) deposits and gathers with the same bilinear weights
    (momentum conserving); "ngp" deposits and gathers nearest-cell;
    "ngp-bilinear" deposits nearest-cell and gathers bilinearly (not an adjoint
    pair, so momentum is not conserved; the original behaviour)."""
    lo = pos.min(0).values - 1e-3
    hi = pos.max(0).values + 1e-3
    h = float((hi - lo).max()) / grid
    fdt = torch.float64 if fft64 else torch.float32
    counts = torch.zeros(grid * grid, device=pos.device, dtype=fdt)
    for i in range(0, pos.shape[0], chunk):
        if mode == "cic":
            for idx, wt in cic_corners((pos[i:i + chunk] - lo) / h - 0.5, grid):
                counts += torch.bincount(idx, weights=wt.to(fdt), minlength=grid * grid).to(fdt)
        else:
            ij = ((pos[i:i + chunk] - lo) / h).floor().long().clamp_(0, grid - 1)
            counts += torch.bincount(ij[:, 0] * grid + ij[:, 1], minlength=grid * grid).to(fdt)
    kx, ky = far_kernel(force_fn, h, radius, grid, pos.device, fft64=fft64)
    c = torch.zeros(2 * grid, 2 * grid, device=pos.device, dtype=fdt)
    c[:grid, :grid] = counts.view(grid, grid)
    fc = torch.fft.rfft2(c)
    fx = torch.fft.irfft2(fc * kx, s=(2 * grid, 2 * grid))[:grid, :grid]
    fy = torch.fft.irfft2(fc * ky, s=(2 * grid, 2 * grid))[:grid, :grid]
    field = torch.stack([fx, fy], dim=-1).reshape(grid * grid, 2).float()
    acc = torch.empty_like(pos)
    for i in range(0, pos.shape[0], chunk):
        if mode == "ngp":
            ij = ((pos[i:i + chunk] - lo) / h).floor().long().clamp_(0, grid - 1)
            acc[i:i + chunk] = field[ij[:, 0] * grid + ij[:, 1]]
            continue
        out = 0
        for idx, wt in cic_corners((pos[i:i + chunk] - lo) / h - 0.5, grid):
            out = out + field[idx] * wt[:, None]
        acc[i:i + chunk] = out
    return acc


@torch.no_grad()
def step(force_fn, pos, vel, dt, radius, strips, far=0, far_mode="cic", far_fft64=True):
    accel = lambda p: tiled_accel(force_fn, p, radius, strips) + (far_accel(force_fn, p, radius, far, mode=far_mode, fft64=far_fft64) if far else 0)
    a0 = accel(pos)
    new_pos = pos + vel * dt + 0.5 * dt * dt * a0
    a1 = accel(new_pos)
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
    ap.add_argument("--clusters", type=int, default=0, help="split --bodies into this many independent clusters on a lattice")
    ap.add_argument("--spacing", type=float, default=300.0)
    ap.add_argument("--strips", type=int, default=100)
    ap.add_argument("--far-grid", type=int, default=0, help="particle-mesh far field on this grid (0 = off, force truncated at --radius)")
    ap.add_argument("--far-fft32", action="store_true", help="float32 mesh counts/kernel/FFT (default float64: fp32 FFT roundoff leaks net force)")
    ap.add_argument("--far-mode", default="cic", choices=["ngp-bilinear", "cic", "ngp"])
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
    side = 0
    if args.clusters:
        pos, vel, spread = init_clusters(args.clusters, args.bodies // args.clusters, args.spacing, dev, args.seed)
        side = int(np.ceil(args.clusters ** 0.5))
        sub_ids = [0, 1, side, side + 1]
        per = args.bodies // args.clusters
        sub = torch.cat([torch.arange(c * per, (c + 1) * per, device=dev) for c in sub_ids])
    else:
        pos, vel, spread = init_state(args.bodies, dev, args.seed)
        sub = torch.arange(0, device=dev)
    sub_frames, sub_ref = [], []
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
            sub_ref.append(ref[0][sub].cpu().numpy())
        sub_frames.append(pos[sub].cpu().numpy())
        np.savez_compressed(args.out, density=np.stack(dens), speed=np.stack(spd), steps=steps, ke=ke, momentum=mom,
                            gap=gap, spread=spread, bodies=args.bodies, radius=args.radius, dt=args.dt,
                            sub=np.stack(sub_frames), sub_ref=np.stack(sub_ref) if sub_ref else np.zeros(0),
                            sub_bodies_per_cluster=args.bodies // max(args.clusters, 1), clusters=args.clusters)

    record(0)
    for k in range(1, args.steps + 1):
        ts = time.time()
        pos, vel = step(force, pos, vel, args.dt, args.radius, args.strips, args.far_grid, args.far_mode, not args.far_fft32)
        if ref is not None:
            ref = step(analytic_force(args.eps), ref[0], ref[1], args.dt, args.radius, args.strips, args.far_grid, args.far_mode, not args.far_fft32)
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
