"""Per-step wall time of KDK rollout steps (force + integrator + audit every 10) for exact / mesh / geo_audit / adaptive (original and optimised).

Usage: PYTHONPATH=. python scripts/scaling_rollout.py --n 100000 --providers exact,mesh,geo_audit,adaptive,adaptive_opt --steps 40 --tag r100k
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import est_train
from scripts import gravity_1b as g1
from scripts import kernels
from scripts import nbody_ic
from scripts import scaling_opt as so
from scripts.adaptive_force import AdaptiveForce

EPS = ao.EPS


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def make(name, args, kernel, head, dev, n):
    idx = torch.arange(n, device=dev)
    if name == "exact":
        return lambda p: ao.exact_accel(p, idx, 1024), None
    if name == "exact_fast":
        return lambda p: so.exact_fast(p, idx, 1024), None
    if name == "mesh":
        fn = g1.analytic_force(EPS)
        return lambda p: (g1.tiled_accel(fn, p.float(), 4.0, 1) + g1.far_accel(fn, p.float(), 4.0, args.grid)).double(), None
    kw = dict(audit_every=10, audit_k=1000, target=0.01, device="cuda")
    if name == "geo_audit":
        af = AdaptiveForce(kernel, head, mode="geo_audit", **kw)
    elif name == "adaptive":
        af = AdaptiveForce(kernel, head, mode="adaptive", **kw)
    elif name == "adaptive_opt":
        af = so.OptForce(kernel, head, mode="adaptive", opts=so.Opts(analytic_grad=True, compile=True), fast_audit=True, **kw)
    elif name == "geo_audit_opt":
        af = so.OptForce(kernel, head, mode="geo_audit", opts=so.Opts(analytic_grad=True, compile=True), fast_audit=True, **kw)
    else:
        raise ValueError(name)
    return af, af


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--ic", default="flyby")
    ap.add_argument("--providers", default="exact,mesh,geo_audit,adaptive,adaptive_opt")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    kernel = kernels.analytic()
    kernels.current = kernel
    head = est_train.load("checkpoints/est_analytic.pt", dev)
    out = {"args": vars(args), "gpu": torch.cuda.get_device_name(), "providers": {}}
    for name in args.providers.split(","):
        torch.manual_seed(4738)
        pos, vel = nbody_ic.IC[args.ic](args.n, kernel, dev)
        n = pos.shape[0]
        force, af = make(name, args, kernel, head, dev, n)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        a = force(pos)
        times = []
        for step in range(args.steps):
            t0 = sync()
            vel = vel + 0.5 * args.dt * a
            pos = pos + args.dt * vel
            a = force(pos)
            vel = vel + 0.5 * args.dt * a
            times.append(sync() - t0)
        w = args.warm
        row = {"N": n, "step_s_median": float(np.median(times[w:])), "step_s_mean": float(np.mean(times[w:])), "step_s_all": times,
               "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "P": float(vel.sum(0).norm())}
        if af is not None:
            st = af.stats
            aud = [s for s in st if s["audit"] is not None]
            row.update({"cost_mean": float(np.mean([s["cost"] for s in st])), "audits": [(s["call"], s["mode"], s["audit"], s.get("audit_time_s")) for s in aud],
                        "audit_time_mean_s": float(np.mean([s["audit_time_s"] for s in aud])) if aud else 0.0,
                        "audit_frac_of_step": float(sum(s["audit_time_s"] for s in aud) / sum(times)) if aud else 0.0,
                        "force_time_median_s": float(np.median([s["time_s"] for s in st[w:]])), "events": af.events,
                        "lam_trace": [s["lam"] for s in st], "mode_trace": [s["mode"] for s in st]})
        out["providers"][name] = row
        print(name, {k: v for k, v in row.items() if k not in ("step_s_all", "lam_trace", "mode_trace")}, flush=True)
        json.dump(out, open(f"results/scaling_{args.tag}.json", "w"))
        del force, af, pos, vel, a
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
