"""Large-N rotating-cluster flyby: model (cutoff 4 + cic far field) for the full run, exact all-pairs
analytic truth for a prefix. Records positions every --record ticks, and energy / |P| / COM every --diag ticks.
Saves full-state checkpoints (pos, vel, tick, frames so far) every --ckpt ticks; --resume continues from one.

Usage: PYTHONPATH=. python3 scripts/gravity_flyby_100k.py --bodies 100000 --steps 10000 --truth-steps 1000 --out results/flyby_100k.npz
"""
import argparse
import os
import time

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from scripts.far_field_accuracy import energy, make_accel
from scripts.gravity_1b import analytic_force, model_force
from scripts.gravity_collision import init_cluster


def diag(pos, vel, eps):
    return [float(energy(pos, vel, eps)), float(vel.double().sum(0).norm()), *[float(x) for x in pos.double().mean(0)]]


@torch.no_grad()
def run(name, accel, pos, vel, steps, dt, record, diag_every, ckpt_every, eps, ckpt_path):
    frames, dg, dg_t, start = [pos.cpu().numpy()], [diag(pos, vel, eps)], [0], 0
    if ckpt_path and os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, weights_only=False)
        pos, vel, start = c["pos"].cuda(), c["vel"].cuda(), c["tick"]
        frames, dg, dg_t = list(c["frames"]), c["diag"], c["diag_t"]
        print(f"{name} resumed at {start}", flush=True)
    a = accel(pos)
    t0 = time.time()
    for k in range(start + 1, steps + 1):
        pos = pos + vel * dt + 0.5 * dt * dt * a
        a1 = accel(pos)
        vel = vel + 0.5 * dt * (a + a1)
        a = a1
        if k % record == 0:
            frames.append(pos.cpu().numpy())
        if k % diag_every == 0:
            dg.append(diag(pos, vel, eps))
            dg_t.append(k)
            print(f"{name} {k} E {dg[-1][0]:.5g} |P| {dg[-1][1]:.4g} com {dg[-1][2]:.2f} {dg[-1][3]:.2f} {(time.time() - t0) / (k - start):.2f}s/step", flush=True)
        if ckpt_path and k % ckpt_every == 0:
            torch.save(dict(pos=pos.cpu(), vel=vel.cpu(), tick=k, frames=frames, diag=dg, diag_t=dg_t), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    return np.stack(frames), np.array(dg), np.array(dg_t), (time.time() - t0) / max(steps - start, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--truth-steps", type=int, default=1000)
    ap.add_argument("--record", type=int, default=40)
    ap.add_argument("--diag", type=int, default=250)
    ap.add_argument("--ckpt", type=int, default=500)
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=60.0)
    ap.add_argument("--separation", type=float, default=800.0)
    ap.add_argument("--impact", type=float, default=240.0)
    ap.add_argument("--approach", type=float, default=11.0)
    ap.add_argument("--spin", type=float, default=9.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="results/flyby_100k.npz")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    half = args.bodies // 2
    ca = torch.tensor([500.0 - args.separation / 2, 500.0 - args.impact / 2], device=dev)
    cb = torch.tensor([500.0 + args.separation / 2, 500.0 + args.impact / 2], device=dev)
    pa, va = init_cluster(half, ca, args.sigma, args.eps, gen, dev)
    pb, vb = init_cluster(half, cb, args.sigma, args.eps, gen, dev)
    for p, v, c in ((pa, va, ca), (pb, vb, cb)):
        rel = p - c
        v += args.spin / args.sigma * torch.stack([-rel[:, 1], rel[:, 0]], dim=1)
    va[:, 0] += args.approach
    vb[:, 0] -= args.approach
    pos, vel = torch.cat([pa, pb]), torch.cat([va, vb])
    print(f"init rms speed {float(vel.norm(dim=1).pow(2).mean().sqrt()):.2f}", flush=True)
    dyn = CentralForceDynamics(dt=args.dt).to(dev)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    out = {}
    os.makedirs("results", exist_ok=True)
    base = args.out.replace(".npz", "")
    model, mdg, mdt, mspt = run("model", make_accel("far", model_force(dyn), 4.0, args.grid), pos.clone(), vel.clone(), args.steps, args.dt, args.record, args.diag, args.ckpt, args.eps, base + "_model_ckpt.pt")
    np.savez_compressed(args.out, model=model, model_diag=mdg, model_diag_t=mdt, dt=args.dt, record=args.record, bodies=args.bodies, half=half, model_s_per_step=mspt)
    print(f"model done {mspt:.2f}s/step", flush=True)
    if args.truth_steps > 0:
        truth, tdg, tdt, tspt = run("truth", make_accel("exact", analytic_force(args.eps), 0, 0), pos.clone(), vel.clone(), args.truth_steps, args.dt, args.record, args.diag, args.ckpt, args.eps, base + "_truth_ckpt.pt")
        np.savez_compressed(args.out, model=model, model_diag=mdg, model_diag_t=mdt, truth=truth, truth_diag=tdg, truth_diag_t=tdt, dt=args.dt, record=args.record, bodies=args.bodies, half=half, model_s_per_step=mspt, truth_s_per_step=tspt)
        print(f"truth done {tspt:.2f}s/step", flush=True)


if __name__ == "__main__":
    main()
