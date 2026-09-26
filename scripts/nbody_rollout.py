"""N-body rollout runner (2D, unit masses, KDK leapfrog, one force evaluation per step), GPU.
Force providers: exact (all-pairs; fast path for the analytic kernel), geo / geo_audit / est / adaptive (scripts/adaptive_force.py,
momentum-symmetric dual tree), bh (target-based Barnes-Hut, analytic only, not momentum-conserving), mesh (uniform particle mesh + cutoff 4
near field, analytic only). Diagnostics every --diag-every steps: energy (exact pair potential of the kernel), |P|, angular momentum about
the centre of mass, COM, radial quantiles. Snapshots every --snap-every steps; with --ref (snapshots of an exact run) it reports
per-snapshot position error, radial-quantile ratios and density-map correlation against it. Full-state checkpoint every --ckpt-every
steps (--resume continues).

Usage: PYTHONPATH=. python scripts/nbody_rollout.py --ic flyby --n 100000 --steps 2000 --force exact --tag flyby_exact
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import est_train
from scripts import gravity_1b as g1
from scripts import kernels
from scripts import nbody_ic
from scripts.adaptive_force import AdaptiveForce, sync

EPS = ao.EPS
QS = (0.25, 0.5, 0.75, 0.9, 0.99)


def pe_analytic(pos, chunk=512):
    p = pos.double()
    s = 0.0
    for i in range(0, len(p), chunk):
        d2 = ((p[None] - p[i:i + chunk][:, None]) ** 2).sum(-1)
        s += float((-(d2 + EPS ** 2) ** -0.5).sum())
    return 0.5 * (s + len(p) * (1 / EPS))


def diagnostics(pos, vel, kernel, step, full_pe=True):
    com = pos.mean(0)
    rel = pos - com
    r = (pos - pos.median(0).values).norm(dim=1)
    ke = 0.5 * float((vel ** 2).sum())
    pe = None
    if full_pe:
        pe = pe_analytic(pos) if kernel.name == "analytic" else kernel.pair_sums(pos, sub=None if len(pos) <= 50000 else 20000)[0]
    return {"step": step, "KE": ke, "PE": pe, "E": None if pe is None else ke + pe, "P": float(vel.sum(0).norm()),
            "L": float((rel[:, 0] * vel[:, 1] - rel[:, 1] * vel[:, 0]).sum()), "com": com.tolist(),
            "rq": [float(r.quantile(q)) for q in QS]}


def make_force(args, kernel, dev, n):
    idx = torch.arange(n, device=dev)
    if args.force == "exact":
        if kernel.name == "analytic":
            return lambda pos: ao.exact_accel(pos, idx, 1024), None
        return lambda pos: kernel.exact_accel(pos, idx, 128), None
    if args.force == "bh":
        def f(pos):
            tr = ao.Tree(pos, 18)
            acc, _, _ = ao.accel_all(tr, idx, args.cap, theta=args.theta)
            a = torch.empty_like(acc)
            a[tr.order] = acc
            return a
        return f, None
    if args.force == "mesh":
        fn = g1.analytic_force(EPS)

        def f(pos):
            p = pos.float()
            return (g1.tiled_accel(fn, p, 4.0, 1) + g1.far_accel(fn, p, 4.0, args.grid)).double()
        return f, None
    head = None
    if args.force in ("est", "adaptive"):
        path = args.est or f"checkpoints/est_{kernel.name}.pt"
        if not os.path.exists(path):
            est_train.save(est_train.train_estimator(kernel, dev), path, kernel.name)
        head = est_train.load(path, dev)
    af = AdaptiveForce(kernel, head, mode=args.force, cap=args.cap, tol=args.tol, target=args.target, audit_every=args.audit_every,
                       audit_k=args.audit_k, geo_theta=args.theta, reprobe_every=args.reprobe_every, lam_max=args.lam_max, device=str(dev))
    if args.lam0 != 1.0:
        af.lam = args.lam0
    return af, af


def density(pos, centre, half, bins=128):
    ij = ((pos - centre + half) / (2 * half) * bins).floor().long()
    ok = (ij >= 0).all(1) & (ij < bins).all(1)
    h = torch.bincount(ij[ok, 0] * bins + ij[ok, 1], minlength=bins * bins).double()
    return h


def vs_ref(snaps, steps, ref, dev):
    out = []
    common = [s for s in steps if s in set(ref["steps"].tolist())]
    for s in common:
        p = torch.from_numpy(snaps[steps.index(s)]).to(dev).double()
        q = torch.from_numpy(ref["pos"][list(ref["steps"]).index(s)]).to(dev).double()
        rr, rp = (q - q.median(0).values).norm(dim=1), (p - p.median(0).values).norm(dim=1)
        half = float(rr.quantile(0.99)) * 1.05
        hp, hq = density(p, q.median(0).values, half), density(q, q.median(0).values, half)
        d = (p - q).norm(dim=1)
        out.append({"step": s, "dx_mean": float(d.mean()), "dx_median": float(d.median()),
                    "rq_ratio": [float(rp.quantile(x) / rr.quantile(x)) for x in QS],
                    "density_corr": float(torch.corrcoef(torch.stack([hp, hq]))[0, 1])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", default="flyby")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--kernel", default="analytic")
    ap.add_argument("--force", default="exact", choices=["exact", "geo", "geo_audit", "est", "adaptive", "bh", "mesh"])
    ap.add_argument("--est", default=None)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--theta", type=float, default=0.35)
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--audit-every", type=int, default=10)
    ap.add_argument("--audit-k", type=int, default=1000)
    ap.add_argument("--reprobe-every", type=int, default=100)
    ap.add_argument("--lam-max", type=float, default=8.0)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--diag-every", type=int, default=100)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    kernel = est_train.make_kernel(args.kernel, dev)
    kernels.current = kernel
    torch.manual_seed(4738)
    pos, vel = nbody_ic.IC[args.ic](args.n, kernel, dev)
    n = pos.shape[0]
    force, af = make_force(args, kernel, dev, n)
    ck = f"results/rollout_{args.tag}_ckpt.pt"
    diags, snaps, snap_steps, step_times, start = [], [], [], [], 0
    if args.resume and os.path.exists(ck):
        blob = torch.load(ck, map_location=dev, weights_only=False)
        pos, vel, start, diags, snaps, snap_steps, step_times = blob["pos"], blob["vel"], blob["step"], blob["diags"], blob["snaps"], blob["snap_steps"], blob["step_times"]
        print("resumed at", start, flush=True)
    a = force(pos)
    diags.append(diagnostics(pos, vel, kernel, start)) if start == 0 else None
    snaps.append(pos.float().cpu().numpy()) if start == 0 else None
    snap_steps.append(0) if start == 0 else None
    t_run = time.time()
    for step in range(start + 1, args.steps + 1):
        t0 = sync()
        vel = vel + 0.5 * args.dt * a
        pos = pos + args.dt * vel
        a = force(pos)
        vel = vel + 0.5 * args.dt * a
        step_times.append(sync() - t0)
        if step % args.diag_every == 0 or step == args.steps:
            d = diagnostics(pos, vel, kernel, step)
            diags.append(d)
            print(f"step {step} E {d['E']} P {d['P']:.3e} L {d['L']:.4e} rq {[round(x, 1) for x in d['rq']]} t/step {np.mean(step_times[-args.diag_every:]):.3f}s", flush=True)
        if step % args.snap_every == 0 or step == args.steps:
            snaps.append(pos.float().cpu().numpy())
            snap_steps.append(step)
        if step % args.ckpt_every == 0:
            torch.save({"pos": pos, "vel": vel, "step": step, "diags": diags, "snaps": snaps, "snap_steps": snap_steps, "step_times": step_times}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    out = {"args": vars(args), "wall_s": time.time() - t_run, "diags": diags, "step_time_mean": float(np.mean(step_times)), "step_time_median": float(np.median(step_times))}
    if af is not None:
        st = af.stats
        out["force"] = {"mean_cost": float(np.mean([s["cost"] for s in st])), "mean_force_time_s": float(np.mean([s["time_s"] for s in st])),
                        "audits": [(s["call"], s["mode"], s["audit"]) for s in st if s["audit"] is not None], "events": af.events,
                        "mode_fraction_est": float(np.mean([s["mode"] == "est" for s in st])), "mean_audit_time_s": float(np.mean([s.get("audit_time_s", 0) for s in st if s["audit"] is not None] or [0])),
                        "lam_trace": [s["lam"] for s in st][::10], "theta_trace": [s["theta"] for s in st][::10], "cost_trace": [s["cost"] for s in st][::10]}
    np.savez_compressed(f"results/rollout_{args.tag}_snaps.npz", pos=np.stack(snaps), steps=np.array(snap_steps))
    if args.ref:
        ref = np.load(args.ref)
        out["vs_ref"] = vs_ref(snaps, snap_steps, {"pos": ref["pos"], "steps": ref["steps"]}, dev)
        for r in out["vs_ref"][:: max(1, len(out["vs_ref"]) // 8)]:
            print("vs_ref", json.dumps(r), flush=True)
    json.dump(out, open(f"results/rollout_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
