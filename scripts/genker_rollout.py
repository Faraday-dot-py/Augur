"""genker: full flyby rollouts for one force law, all providers on the identical IC (virialised from --ic-kernel, default the kernel itself).
Providers: exact, exact_pert (1e-6 position perturbation = chaos floor), exact_analytic (analytic-force exact rollout, same IC),
adaptive_own / adaptive_transfer (AdaptiveForce with the kernel's own / the analytic-trained estimator), geo_audit, est_own / est_transfer
(estimator, no audit controller). Every tree provider also gets an independent audit (fresh generator, --check-k particles vs exact) every
--check-every steps. KDK leapfrog dt 0.05, one force eval per step. Diagnostics reuse scripts/nbody_rollout.py.

Usage: PYTHONPATH=. python scripts/genker_rollout.py --kernel lj --n 20000 --steps 1000 --tag lj
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import est_train
from scripts import genker_kernels as gk
from scripts import kernels
from scripts import nbody_ic
from scripts import nbody_rollout as nr
from scripts.adaptive_force import AdaptiveForce, sync


def build(prov, kernel, ana, dev, n, args):
    idx = torch.arange(n, device=dev)
    if prov in ("exact", "exact_pert"):
        return (lambda pos: kernel.exact_accel(pos, idx, args.chunk)), None, kernel
    if prov == "exact_analytic":
        return (lambda pos: ao.exact_accel(pos, idx, 1024)), None, ana
    mode = {"adaptive_own": "adaptive", "adaptive_transfer": "adaptive", "geo_audit": "geo_audit", "est_own": "est", "est_transfer": "est"}[prov]
    head = None
    if prov.endswith("own"):
        head = est_train.load(f"checkpoints/genker_est_{kernel.name}.pt", dev)
    elif prov.endswith("transfer"):
        head = est_train.load(args.analytic_est, dev)
    af = AdaptiveForce(kernel, head, mode=mode, target=args.target, audit_every=args.audit_every, audit_k=args.audit_k, device=str(dev))
    gen = torch.Generator(device=dev).manual_seed(999)
    calls = [0]

    def f(pos):
        kernels.current = kernel
        a = af(pos)
        if calls[0] % args.check_every == 0:
            err = de.audit_e2e(pos, a, args.check_k, gen)
            af.stats[-1]["check"] = err
        calls[0] += 1
        return a
    return f, af, kernel


def run(prov, force, af, kern, pos0, vel0, args):
    pos, vel = pos0.clone(), vel0.clone()
    if prov == "exact_pert":
        pos = pos + 1e-6 * torch.randn(pos.shape, generator=torch.Generator(device=pos.device).manual_seed(4738), device=pos.device, dtype=pos.dtype)
    kernels.current = kern
    diags = [nr.diagnostics(pos, vel, kern, 0)]
    snaps, snap_steps, times = [pos.float().cpu().numpy()], [0], []
    a = force(pos)
    t_run = time.time()
    for step in range(1, args.steps + 1):
        t0 = sync()
        vel = vel + 0.5 * args.dt * a
        pos = pos + args.dt * vel
        a = force(pos)
        vel = vel + 0.5 * args.dt * a
        times.append(sync() - t0)
        if step % args.diag_every == 0 or step == args.steps:
            d = nr.diagnostics(pos, vel, kern, step)
            diags.append(d)
            print(prov, "step", step, f"E {d['E']:.6e} P {d['P']:.2e} L {d['L']:.4e} t/step {np.mean(times[-args.diag_every:]):.3f}", flush=True)
        if step % args.snap_every == 0 or step == args.steps:
            snaps.append(pos.float().cpu().numpy())
            snap_steps.append(step)
    out = {"wall_s": time.time() - t_run, "step_time_mean": float(np.mean(times)), "step_time_median": float(np.median(times)), "diags": diags,
           "final_finite": bool(torch.isfinite(pos).all())}
    if af is not None:
        st = af.stats
        chk = [s["check"] for s in st if "check" in s]
        out["force"] = {"mean_cost": float(np.mean([s["cost"] for s in st])), "mean_force_time_s": float(np.mean([s["time_s"] for s in st])),
                        "mean_audit_time_s": float(np.mean([s.get("audit_time_s", 0) for s in st if s["audit"] is not None] or [0])),
                        "mode_fraction_est": float(np.mean([s["mode"] == "est" for s in st])), "events": af.events,
                        "audits": [(s["call"], s["mode"], s["audit"]) for s in st if s["audit"] is not None],
                        "check_trace": chk, "check_max": float(max(chk)), "check_mean": float(np.mean(chk)),
                        "lam_trace": [s["lam"] for s in st][::10], "theta_trace": [s["theta"] for s in st][::10], "cost_trace": [s["cost"] for s in st][::10]}
    np.savez_compressed(f"results/genker_{args.tag}_{prov}_snaps.npz", pos=np.stack(snaps), steps=np.array(snap_steps))
    return out, snaps, snap_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--ic-kernel", default=None)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--providers", default="exact,exact_pert,adaptive_own,adaptive_transfer,geo_audit,est_own,est_transfer")
    ap.add_argument("--analytic-est", default="checkpoints/est_analytic.pt")
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--audit-every", type=int, default=10)
    ap.add_argument("--audit-k", type=int, default=1000)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--check-k", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--diag-every", type=int, default=100)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    kernel = gk.make_kernel(args.kernel, dev)
    ana = kernels.analytic()
    kernels.current = kernel
    torch.manual_seed(4738)
    ick = gk.make_kernel(args.ic_kernel, dev) if args.ic_kernel else kernel
    pos0, vel0 = nbody_ic.IC["flyby"](args.n, ick, dev)
    print("IC virialised from", ick.name, "vel rms", float(vel0.norm(dim=1).pow(2).mean().sqrt()), flush=True)
    res = {"args": vars(args), "prov": {}}
    ref = None
    for prov in args.providers.split(","):
        force, af, kern = build(prov, kernel, ana, dev, args.n, args)
        out, snaps, steps = run(prov, force, af, kern, pos0, vel0, args)
        if prov == "exact":
            ref = {"pos": np.stack(snaps), "steps": np.array(steps)}
        elif ref is not None:
            out["vs_exact"] = nr.vs_ref(snaps, steps, ref, dev)
        res["prov"][prov] = out
        json.dump(res, open(f"results/genker_{args.tag}.json", "w"))
        print(prov, "done", f"{out['wall_s']:.0f}s", json.dumps(out.get("force", {}).get("events", [])[:3]), flush=True)


if __name__ == "__main__":
    main()
