"""3D two-cluster bound binary (unit masses, softened 1/r^2 gravity, KDK leapfrog, exact all-pairs force), GPU.
Same setup as orbit_bh.py --mode binary but positions/velocities are (N, 3): two n/2 3D Gaussian clusters (sigma --sigma) on a circular
relative orbit in the x-y plane at separation --d; --tilt rotates the orbit plane about x by that many degrees (z then matters in the orbit);
--vcap-frac caps each body's initial internal speed at that fraction of its own-cluster escape speed. Internal dispersion from the 3D virial
theorem 3 N s^2 = sum_{i<j} d f(d) (subsampled). Tracks the same retention/separation keys as orbit_bh.py (orbit_summary.py works),
"L" is |L| of the whole system, "Lz" its z component. Full-state checkpoint every --ckpt steps (--resume continues).

Usage: PYTHONPATH=. python scripts/orbit_3d.py --steps 130000 --tag binary3d
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

EPS = 0.5


def accel(pos, chunk):
    n = len(pos)
    out = torch.empty_like(pos)
    for i in range(0, n, chunk):
        d = pos[None] - pos[i:i + chunk][:, None]
        out[i:i + chunk] = (d * ((d * d).sum(-1) + EPS ** 2).pow(-1.5)[..., None]).sum(1)
    return out


def potentials(pos, chunk=1024):
    p = pos.double()
    out = torch.empty(len(p), dtype=torch.float64, device=p.device)
    for i in range(0, len(p), chunk):
        d2 = ((p[None] - p[i:i + chunk][:, None]) ** 2).sum(-1)
        out[i:i + chunk] = -((d2 + EPS ** 2) ** -0.5).sum(1) + 1 / EPS
    return out


def virial_sigma(pos, ratio=1.0, sub=6000):
    n = len(pos)
    p = pos[torch.randperm(n, device=pos.device)[:sub]]
    d2 = ((p[None] - p[:, None]) ** 2).sum(-1)
    w = (d2 / (d2 + EPS ** 2) ** 1.5).sum() / 2 * (n / len(p)) ** 2
    return math.sqrt(ratio * float(w) / (3 * n))


def cluster(n, centre, sigma, gen, dev):
    pos = torch.randn(n, 3, generator=gen, device=dev, dtype=torch.float64) * sigma + torch.tensor(centre, device=dev, dtype=torch.float64)
    return pos, torch.randn(n, 3, generator=gen, device=dev, dtype=torch.float64) * virial_sigma(pos)


def bound_cap(p, v, frac):
    vesc = (2 * (-potentials(p)).clamp(min=0)).sqrt()
    return v * (frac * vesc / v.norm(dim=1).clamp(min=1e-12)).clamp(max=1.0)[:, None]


def med(x):
    return x.median(0).values


def build(args, dev):
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    h = args.n // 2
    pa, va = cluster(h, (-args.d / 2, 0.0, 0.0), args.sigma, gen, dev)
    pb, vb = cluster(h, (args.d / 2, 0.0, 0.0), args.sigma, gen, dev)
    vrel = args.vfac * math.sqrt(args.n * args.d ** 2 / (args.d ** 2 + EPS ** 2) ** 1.5)
    if args.vcap_frac > 0:
        va, vb = bound_cap(pa, va, args.vcap_frac), bound_cap(pb, vb, args.vcap_frac)
    th = math.radians(args.tilt)
    u = torch.tensor([0.0, math.cos(th), math.sin(th)], device=dev, dtype=torch.float64)
    va, vb = va - vrel / 2 * u, vb + vrel / 2 * u
    print(f"binary3d d {args.d} sigma {args.sigma} tilt {args.tilt} vrel {vrel:.3f} period {2 * math.pi * args.d / vrel:.1f} = {2 * math.pi * args.d / vrel / args.dt:.0f} steps", flush=True)
    pos, vel = torch.cat([pa, pb]), torch.cat([va, vb])
    return pos, vel - vel.mean(0)


def track_binary(pos, args):
    h = args.n // 2
    ca, cb = med(pos[:h]), med(pos[h:])
    ra, rb = (pos[:h] - ca).norm(dim=1), (pos[h:] - cb).norm(dim=1)
    xa, xb = (pos[:h] - cb).norm(dim=1), (pos[h:] - ca).norm(dim=1)
    s = args.sigma
    return {"sep": float((ca - cb).norm()), "ca": ca.tolist(), "cb": cb.tolist(),
            "a_r4": float((ra < 4 * s).double().mean()), "b_r4": float((rb < 4 * s).double().mean()),
            "a_r8": float((ra < 8 * s).double().mean()), "b_r8": float((rb < 8 * s).double().mean()),
            "a_own": float((ra < xa).double().mean()), "b_own": float((rb < xb).double().mean())}


def diagnostics(pos, vel, args, step):
    vcm = vel.mean(0)
    phi = potentials(pos)
    ke_i = 0.5 * ((vel - vcm) ** 2).sum(1)
    ei = ke_i + phi
    rel = pos - pos.mean(0)
    L = torch.cross(rel, vel, dim=1).sum(0)
    r = (pos - med(pos)).norm(dim=1)
    h = args.n // 2
    return {"step": step, "KE": float(ke_i.sum()), "PE": float(0.5 * phi.sum()), "E": float(ke_i.sum() + 0.5 * phi.sum()),
            "P": float(vel.sum(0).norm()), "L": float(L.norm()), "Lz": float(L[2]),
            "frac_E_pos": float((ei > 0).double().mean()), "rq": [float(r.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)],
            "clamped_frac": 0.0, "frac_E_pos_a": float((ei[:h] > 0).double().mean()), "frac_E_pos_b": float((ei[h:] > 0).double().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=130000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=20.0)
    ap.add_argument("--d", type=float, default=600.0)
    ap.add_argument("--vfac", type=float, default=1.0)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--vcap-frac", type=float, default=0.75)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--record", type=int, default=500)
    ap.add_argument("--track", type=int, default=200)
    ap.add_argument("--diag", type=int, default=5000)
    ap.add_argument("--ckpt", type=int, default=10000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    pos, vel = build(args, dev)
    ck = f"results/orbit_{args.tag}_ckpt.pt"
    frames, fsteps, diags, tracks, tsteps, start = [pos.float().cpu().numpy()], [0], [], [], [], 0
    if args.resume and os.path.exists(ck):
        b = torch.load(ck, map_location=dev, weights_only=False)
        pos, vel, start, frames, fsteps, diags, tracks, tsteps = b["pos"], b["vel"], b["step"], b["frames"], b["fsteps"], b["diags"], b["tracks"], b["tsteps"]
        print("resumed at", start, flush=True)
    if start == 0:
        diags.append(diagnostics(pos, vel, args, 0))
        print(json.dumps(diags[-1]), flush=True)
        tracks.append(track_binary(pos, args))
        tsteps.append(0)
    a = accel(pos.float(), args.chunk).double()
    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        vel = vel + 0.5 * args.dt * a
        pos = pos + args.dt * vel
        a = accel(pos.float(), args.chunk).double()
        vel = vel + 0.5 * args.dt * a
        if step % args.track == 0:
            tracks.append(track_binary(pos, args))
            tsteps.append(step)
        if step % args.record == 0:
            frames.append(pos.float().cpu().numpy())
            fsteps.append(step)
        if step % args.diag == 0 or step == args.steps:
            d = diagnostics(pos, vel, args, step)
            diags.append(d)
            print(f"step {step} E {d['E']:.6g} P {d['P']:.2e} L {d['L']:.5g} Lz {d['Lz']:.5g} Epos {d['frac_E_pos']:.5f} rq {[round(x, 1) for x in d['rq']]} sep {tracks[-1]['sep']:.1f} a_r8 {tracks[-1]['a_r8']:.4f} b_r8 {tracks[-1]['b_r8']:.4f} {(time.time() - t0) / (step - start):.3f}s/step", flush=True)
        if step % args.ckpt == 0:
            torch.save({"pos": pos, "vel": vel, "step": step, "frames": frames, "fsteps": fsteps, "diags": diags, "tracks": tracks, "tsteps": tsteps}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    np.savez_compressed(f"results/orbit_{args.tag}_snaps.npz", pos=np.stack(frames), steps=np.array(fsteps))
    json.dump({"args": vars(args), "wall_s": time.time() - t0, "diags": diags, "tracks": tracks, "track_steps": tsteps}, open(f"results/orbit_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
