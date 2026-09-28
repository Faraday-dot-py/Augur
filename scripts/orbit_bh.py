"""Two-cluster bound binary and speed-limited "black hole" N-body runs (2D, unit masses, softened 1/r^2 gravity, KDK leapfrog), GPU.
binary: two n/2 Gaussian clusters (sigma --sigma) on a circular relative orbit at separation --d (--vfac scales the circular speed; --vcap-frac f caps each body's initial internal speed at f x its own-cluster escape speed so no body starts unbound);
  tracks each cluster's mass retention (fraction of its own bodies within 4/8 sigma of its own median centre, and closer to its own
  centre than the other's), COM separation, and the fraction of bodies with positive energy in the system frame.
blackhole: one Gaussian cluster; --c is a speed limit (bodies are clamped to |v| <= c after every step). With horizon
  R_s = 2 N / c^2 (softened potential depth ~ N / r), a body at r < R_s has |Phi| > c^2 / 2, so E < 0 for every allowed speed and it cannot
  escape; reports the fraction beyond R_s, the fraction with E > 0, and clamp counts. --c 0 is the uncapped control.
--kernel learned replaces the analytic pair force by the learned CentralForceDynamics f(d) (checkpoints/gravity_central_v1.pt, --force exact only; diagnostics stay analytic). The MLP is tabulated on a 2^20-point log-d grid and linearly interpolated (error vs direct evaluation printed by bench_learned_force.py).
--spill writes recorded frames to results/orbit_<tag>_frames_<step>.npy at each checkpoint instead of keeping them in memory / the checkpoint (for fine --record under a small host-RAM cap); the final npz is assembled from those chunks.
--relativistic (with --c) replaces the clamp by relativistic kinetics: the state is momentum p (dp/dt = F), v = p / sqrt(1 + p^2/c^2), KE = c^2 (sqrt(1 + p^2/c^2) - 1);
  same leapfrog, |v| < c by construction, energy and momentum conserved.
Force: exact all-pairs or mesh (cutoff 4 near field + cic far field, grid --grid; --box-q q sets the far-field box from the q / 1-q position quantiles, far-field positions clamped to it, so escapers do not coarsen the grid). Full-state checkpoint every --ckpt steps (--resume continues).

Usage: PYTHONPATH=. python scripts/orbit_bh.py --mode binary --steps 18000 --force mesh --tag binary_mesh
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import est_train
from scripts import gravity_1b as g1
from scripts import kernels
from scripts import nbody_ic
from model.central_force import CentralForceDynamics

EPS = ao.EPS


def potentials(pos, chunk=1024):
    p = pos.double()
    out = torch.empty(len(p), dtype=torch.float64, device=p.device)
    for i in range(0, len(p), chunk):
        d2 = ((p[None] - p[i:i + chunk][:, None]) ** 2).sum(-1)
        out[i:i + chunk] = -((d2 + EPS ** 2) ** -0.5).sum(1) + 1 / EPS
    return out


def bound_cap(p, v, frac):
    vesc = (2 * (-potentials(p)).clamp(min=0)).sqrt()
    return v * (frac * vesc / v.norm(dim=1).clamp(min=1e-12)).clamp(max=1.0)[:, None]


def med(x):
    return x.median(0).values


def build(args, kernel, dev):
    gen = nbody_ic._gen(dev, args.seed)
    if args.mode == "binary":
        h = args.n // 2
        pa, va = nbody_ic._cluster(h, (-args.d / 2, 0.0), args.sigma, kernel, gen, dev)
        pb, vb = nbody_ic._cluster(h, (args.d / 2, 0.0), args.sigma, kernel, gen, dev)
        vrel = args.vfac * math.sqrt(args.n * args.d ** 2 / (args.d ** 2 + EPS ** 2) ** 1.5)
        if args.vcap_frac > 0:
            va, vb = bound_cap(pa, va, args.vcap_frac), bound_cap(pb, vb, args.vcap_frac)
        va[:, 1] -= vrel / 2
        vb[:, 1] += vrel / 2
        print(f"binary d {args.d} sigma {args.sigma} vrel {vrel:.3f} period {2 * math.pi * args.d / vrel:.1f} = {2 * math.pi * args.d / vrel / args.dt:.0f} steps", flush=True)
        return nbody_ic._finish(torch.cat([pa, pb]), torch.cat([va, vb]))
    return nbody_ic._finish(*nbody_ic._cluster(args.n, (0.0, 0.0), args.sigma, kernel, gen, dev, ratio=args.ratio))


def learned_table_fn(dyn, dev, m=1 << 20, dmin=1e-6):
    lo, hi = math.log(dmin), math.log(g1.FORCE_TRAIN_MAX)
    h = dyn.force(torch.linspace(lo, hi, m, device=dev)[:, None])[:, 0]
    step = (hi - lo) / (m - 1)

    def fn(d):
        t = ((d.log() - lo) / step).clamp(0, m - 1)
        i = t.floor().long().clamp(max=m - 2)
        w = (t - i).clamp(0, 1)
        return (h[i] * (1 - w) + h[i + 1] * w) / (d ** 2 + 1.0)
    return fn


def learned_accel(pos, fn, chunk):
    p = pos.float()
    out = torch.empty_like(p)
    for i in range(0, len(p), chunk):
        rel = p[None] - p[i:i + chunk][:, None]
        d = (rel ** 2).sum(-1, keepdim=True).add(1e-12).sqrt()
        out[i:i + chunk] = (fn(d) * rel / d).sum(1)
    return out.double()


def make_force(args, n, dev):
    if args.kernel == "learned":
        dyn = CentralForceDynamics(dt=0.1).to(dev)
        dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
        dyn.eval()
        fn = learned_table_fn(dyn, dev)
        return lambda pos: learned_accel(pos, fn, args.lchunk)
    if args.force == "exact":
        idx = torch.arange(n, device=dev)
        return lambda pos: ao.exact_accel(pos, idx, 1024)
    fn = g1.analytic_force(EPS)
    if args.box_q > 0:
        def f(pos):
            p = pos.float()
            lo, hi = torch.quantile(p, 1 - args.box_q, dim=0), torch.quantile(p, args.box_q, dim=0)
            return (g1.tiled_accel(fn, p, 4.0, 1) + g1.far_accel(fn, torch.maximum(torch.minimum(p, hi), lo), 4.0, args.grid)).double()
        return f
    return lambda pos: (g1.tiled_accel(fn, pos.float(), 4.0, 1) + g1.far_accel(fn, pos.float(), 4.0, args.grid)).double()


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


def speed(p, args):
    return p / (1 + (p ** 2).sum(1, keepdim=True) / args.c ** 2).sqrt() if args.relativistic else p


def diagnostics(pos, vel, args, step, clamped):
    vcm = vel.mean(0)
    phi = potentials(pos)
    ke_i = args.c ** 2 * ((1 + ((vel - vcm) ** 2).sum(1) / args.c ** 2).sqrt() - 1) if args.relativistic else 0.5 * ((vel - vcm) ** 2).sum(1)
    ei = ke_i + phi
    com = pos.mean(0)
    rel = pos - com
    r = (pos - med(pos)).norm(dim=1)
    d = {"step": step, "KE": float(ke_i.sum()), "PE": float(0.5 * phi.sum()), "E": float(ke_i.sum() + 0.5 * phi.sum()),
         "P": float(vel.sum(0).norm()), "L": float((rel[:, 0] * vel[:, 1] - rel[:, 1] * vel[:, 0]).sum()),
         "frac_E_pos": float((ei > 0).double().mean()), "rq": [float(r.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)],
         "clamped_frac": clamped}
    if args.mode == "binary":
        h = args.n // 2
        d["frac_E_pos_a"], d["frac_E_pos_b"] = float((ei[:h] > 0).double().mean()), float((ei[h:] > 0).double().mean())
    if args.c > 0:
        rs = 2 * args.n / args.c ** 2
        d["R_s"], d["frac_beyond_Rs"] = rs, float((r > rs).double().mean())
        d["max_speed"] = float(speed(vel, args).norm(dim=1).max())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "blackhole"], required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--force", choices=["exact", "mesh"], default="mesh")
    ap.add_argument("--kernel", choices=["analytic", "learned"], default="analytic")
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--lchunk", type=int, default=512)
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=40.0)
    ap.add_argument("--d", type=float, default=600.0)
    ap.add_argument("--vfac", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.0)
    ap.add_argument("--relativistic", action="store_true")
    ap.add_argument("--vcap-frac", type=float, default=0.0)
    ap.add_argument("--box-q", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--record", type=int, default=50)
    ap.add_argument("--track", type=int, default=20)
    ap.add_argument("--diag", type=int, default=250)
    ap.add_argument("--spill", action="store_true")
    ap.add_argument("--ckpt", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    kernel = est_train.make_kernel("analytic", dev)
    torch.set_grad_enabled(False)
    kernels.current = kernel
    torch.manual_seed(args.seed)
    pos, vel = build(args, kernel, dev)
    if args.relativistic:
        vs = vel.norm(dim=1, keepdim=True)
        vel = vel * (1 / (1 - (vs / args.c).clamp(max=0.99) ** 2).sqrt()) * (vs.clamp(max=0.99 * args.c) / vs.clamp(min=1e-12))
        vel = vel - vel.mean(0)
    force = make_force(args, pos.shape[0], dev)
    ck = f"results/orbit_{args.tag}_ckpt.pt"
    frames, fsteps, diags, tracks, tsteps, start, nclamp = [pos.float().cpu().numpy()], [0], [], [], [], 0, 0
    if args.resume and os.path.exists(ck):
        b = torch.load(ck, map_location=dev, weights_only=False)
        pos, vel, start, frames, fsteps, diags, tracks, tsteps = b["pos"], b["vel"], b["step"], b["frames"], b["fsteps"], b["diags"], b["tracks"], b["tsteps"]
        print("resumed at", start, flush=True)
        if args.spill:
            for f in glob.glob(f"results/orbit_{args.tag}_frames_*.npy"):
                if int(f[-11:-4]) > start:
                    os.remove(f)
    if start == 0:
        diags.append(diagnostics(pos, vel, args, 0, 0.0))
        print(json.dumps(diags[-1]), flush=True)
        if args.mode == "binary":
            tracks.append(track_binary(pos, args))
            tsteps.append(0)
    a = force(pos)
    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        vel = vel + 0.5 * args.dt * a
        pos = pos + args.dt * speed(vel, args)
        a = force(pos)
        vel = vel + 0.5 * args.dt * a
        if args.c > 0 and not args.relativistic:
            sp = vel.norm(dim=1)
            over = sp > args.c
            nclamp += int(over.sum())
            vel = torch.where(over[:, None], vel * (args.c / sp.clamp(min=1e-12))[:, None], vel)
        if args.mode == "binary" and step % args.track == 0:
            tracks.append(track_binary(pos, args))
            tsteps.append(step)
        if step % args.record == 0:
            frames.append(pos.float().cpu().numpy())
            fsteps.append(step)
        if step % args.diag == 0 or step == args.steps:
            d = diagnostics(pos, vel, args, step, nclamp / (pos.shape[0] * args.diag))
            nclamp = 0
            diags.append(d)
            extra = f" sep {tracks[-1]['sep']:.1f} a_r8 {tracks[-1]['a_r8']:.4f} b_r8 {tracks[-1]['b_r8']:.4f}" if tracks else ""
            print(f"step {step} E {d['E']:.6g} P {d['P']:.2e} L {d['L']:.5g} Epos {d['frac_E_pos']:.5f} rq {[round(x, 1) for x in d['rq']]} clamp {d['clamped_frac']:.3f}{extra} {(time.time() - t0) / (step - start):.3f}s/step", flush=True)
        if step % args.ckpt == 0:
            if args.spill:
                np.save(f"results/orbit_{args.tag}_frames_{step:07d}.npy", np.stack(frames))
                frames = []
            torch.save({"pos": pos, "vel": vel, "step": step, "frames": frames, "fsteps": fsteps, "diags": diags, "tracks": tracks, "tsteps": tsteps}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    if args.spill:
        parts = [np.load(f) for f in sorted(glob.glob(f"results/orbit_{args.tag}_frames_*.npy"))] + ([np.stack(frames)] if frames else [])
        allf = np.concatenate(parts)
        del parts
        np.savez(f"results/orbit_{args.tag}_snaps.npz", pos=allf, steps=np.array(fsteps))
    else:
        np.savez_compressed(f"results/orbit_{args.tag}_snaps.npz", pos=np.stack(frames), steps=np.array(fsteps))
    json.dump({"args": vars(args), "wall_s": time.time() - t0, "diags": diags, "tracks": tracks, "track_steps": tsteps}, open(f"results/orbit_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
