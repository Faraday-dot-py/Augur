"""H5/H2 wall-clock scaling: one force evaluation vs N for ALF est, geo, geo at matched error, target BH, mesh, exact (analytic kernel).

Usage: PYTHONPATH=. python scripts/scaling_bench.py --states uniform,flyby --ns 10000,30000,100000 --tag a
"""
import argparse
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import est_train
from scripts import gravity_1b as g1
from scripts import kernels
from scripts import nbody_ic
from scripts import scaling_timed as st
from scripts.adaptive_force import AdaptiveForce

EPS = ao.EPS


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def timeit(fn, reps, warm=1):
    for _ in range(warm):
        fn()
    ts, out = [], None
    for _ in range(reps):
        t0 = sync()
        out = fn()
        ts.append(sync() - t0)
    return float(np.median(ts)), ts, out


def peak_gb(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30, out


def build(state, n, dev, kernel):
    if state == "t10000":
        return ao.load_state("flyby_t10000", SimpleNamespace(npz="results/flyby_100k.npz", bodies=n), dev)
    return nbody_ic.IC[state](n, kernel, dev)[0]


def err(a, S, a_ex):
    return ao.metrics(a[S], a_ex)


def calib(fn, lo, hi, target, iters=9):
    """Bisection in log space on a monotone-increasing error(param); returns (param, err, extra) closest to target."""
    best = None
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        e, extra = fn(mid)
        if best is None or abs(math.log(e / target)) < abs(math.log(best[1] / target)):
            best = (mid, e, extra)
        if e > target:
            hi = mid
        else:
            lo = mid
    return best


def stats_row(ts):
    return {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "max_s": float(np.max(ts)), "reps": len(ts)}


def run(state, n, args, dev, kernel, head, out):
    r = {"state": state, "N": n}
    pos = build(state, n, dev, kernel)
    n = pos.shape[0]
    r["N"] = n
    S = torch.randperm(n, generator=torch.Generator().manual_seed(4738))[:args.k_ref].to(dev)
    a_ex = ao.exact_accel(pos, S)
    reps = args.reps
    print(f"== {state} N={n}", flush=True)

    def new_af(mode, **kw):
        return AdaptiveForce(kernel, head, mode=mode, device="cuda", **kw)

    af = new_af("est")
    t0 = sync()
    a = af(pos)
    r["est_cold_s"] = sync() - t0
    a = af(pos)
    r["est_default"] = {"lam": 1.0, "cost": af.stats[-1]["cost"], **err(a, S, a_ex)}
    print("est default", r["est_default"], flush=True)

    def est_err(lam):
        af.lam = lam
        a = af(pos)
        return err(a, S, a_ex)["rel_l2"], af.stats[-1]["cost"]

    lam, e_est, cost = calib(est_err, 0.05, 64.0, args.target)
    af.lam = lam
    af(pos)
    t_est, ts, a = timeit(lambda: af(pos), reps)
    mem, _ = peak_gb(lambda: af(pos))
    r["est"] = {"lam": lam, **stats_row(ts), "peak_gb": mem, "cost": af.stats[-1]["cost"], **err(a, S, a_ex), "net_ratio": ao.net_ratio(a)}
    print("est", r["est"], flush=True)
    tim = []
    for _ in range(reps + 1):
        T = st.Tm()
        _, info = st.timed_call(pos, af, T, "est")
        tim.append(dict(T.d))
    keys = tim[0].keys()
    r["est_breakdown_s"] = {k: float(np.median([t[k] for t in tim[1:]])) for k in keys}
    r["est_info"] = info
    print("est breakdown", r["est_breakdown_s"], info, flush=True)
    ta = []
    gen = torch.Generator(device=dev).manual_seed(4738)
    de.audit_e2e(pos, af.prev_a, 1000, gen)
    for _ in range(reps):
        t0 = sync()
        de.audit_e2e(pos, af.prev_a, 1000, gen)
        ta.append(sync() - t0)
    r["audit_k1000_s"] = stats_row(ta)
    e_match = r["est"]["rel_l2"]
    del af

    def geo_at(theta):
        g = new_af("geo", geo_theta=theta)
        g(pos)
        t_, ts_, a_ = timeit(lambda: g(pos), reps)
        mem_, _ = peak_gb(lambda: g(pos))
        return g, a_, ts_, mem_

    g, a, ts, mem = geo_at(0.35)
    r["geo035"] = {"theta": 0.35, **stats_row(ts), "peak_gb": mem, "cost": g.stats[-1]["cost"], **err(a, S, a_ex), "net_ratio": ao.net_ratio(a)}
    print("geo 0.35", r["geo035"], flush=True)
    tim = []
    for _ in range(reps + 1):
        T = st.Tm()
        _, info = st.timed_call(pos, g, T, "geo")
        tim.append(dict(T.d))
    r["geo035_breakdown_s"] = {k: float(np.median([t[k] for t in tim[1:]])) for k in tim[0]}
    del g

    def geo_err(theta):
        g = new_af("geo", geo_theta=theta)
        a = g(pos)
        return err(a, S, a_ex)["rel_l2"], g.stats[-1]["cost"]

    th, e_g, c_g = calib(geo_err, 0.1, 1.2, e_match)
    g, a, ts, mem = geo_at(th)
    r["geo_matched"] = {"theta": th, "target_rel_l2": e_match, **stats_row(ts), "peak_gb": mem, "cost": g.stats[-1]["cost"], **err(a, S, a_ex), "net_ratio": ao.net_ratio(a)}
    print("geo matched", r["geo_matched"], flush=True)
    del g
    json.dump(out, open(args.out, "w"))

    if n <= args.bh_max:
        idx = torch.arange(n, device=dev)
        best = None
        for th in (0.2, 0.3, 0.4, 0.5, 0.7):
            tr = ao.Tree(pos, 18)
            acc, nm, ns = ao.accel_all(tr, tr.inv[S], 8, theta=th)
            e = ao.metrics(acc, a_ex)["rel_l2"]
            cost = float((nm + ns).mean())
            if best is None or abs(math.log(e / e_match)) < abs(math.log(best[1] / e_match)):
                best = (th, e, cost)
        th = best[0]

        def bh_all():
            tr = ao.Tree(pos, 18)
            acc, nm, ns = ao.accel_all(tr, idx, 8, theta=th)
            a = torch.empty_like(acc)
            a[tr.order] = acc
            return a, float((nm + ns).mean())

        t0 = sync()
        bh_all()
        one = sync() - t0
        if one < args.max_call_s:
            t_bh, ts, (a, cost) = timeit(bh_all, min(reps, 3), warm=0)
            mem, _ = peak_gb(bh_all)
            r["bh"] = {"theta": th, **stats_row(ts), "peak_gb": mem, "cost": cost, **err(a, S, a_ex), "net_ratio": ao.net_ratio(a)}
        else:
            r["bh"] = {"theta": th, "skipped_first_call_s": one, "cost": best[2], "rel_l2": best[1]}
        print("bh", r["bh"], flush=True)

    mesh_rows = []
    fn = g1.analytic_force(EPS)
    p32 = pos.float()
    for grid in args.grids:
        try:
            def mesh():
                return (g1.tiled_accel(fn, p32, 4.0, 1) + g1.far_accel(fn, p32, 4.0, grid)).double()
            a = mesh()
            e = err(a, S, a_ex)
            t_m, ts, _ = timeit(mesh, min(reps, 3), warm=0)
            mem, _ = peak_gb(mesh)
            mesh_rows.append({"grid": grid, **stats_row(ts), "peak_gb": mem, **e, "net_ratio": ao.net_ratio(a)})
            print("mesh", mesh_rows[-1], flush=True)
            del a
            if e["rel_l2"] < 0.02:
                break
        except torch.cuda.OutOfMemoryError:
            mesh_rows.append({"grid": grid, "oom": True})
            torch.cuda.empty_cache()
            break
    r["mesh"] = mesh_rows
    json.dump(out, open(args.out, "w"))

    if n <= args.exact_max:
        idx = torch.arange(n, device=dev)
        fex = lambda: ao.exact_accel(pos, idx, 1024)
        t_ex, ts, _ = timeit(fex, min(reps, 3 if n <= 100000 else 1), warm=1 if n <= 100000 else 0)
        mem, _ = peak_gb(fex) if n <= 100000 else (float("nan"), None)
        r["exact"] = {**stats_row(ts), "peak_gb": mem, "pairs_per_particle": n}
        print("exact", r["exact"], flush=True)
    out["rows"].append(r)
    json.dump(out, open(args.out, "w"))
    del pos, a_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="uniform,flyby")
    ap.add_argument("--ns", default="10000,30000,100000")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--k-ref", type=int, default=2000)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--grids", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--bh-max", type=int, default=300000)
    ap.add_argument("--exact-max", type=int, default=300000)
    ap.add_argument("--max-call-s", type=float, default=60.0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    args.out = f"results/scaling_{args.tag}.json"
    dev = torch.device("cuda")
    kernel = kernels.analytic()
    kernels.current = kernel
    torch.manual_seed(4738)
    head = est_train.load("checkpoints/est_analytic.pt", dev)
    out = {"args": vars(args), "gpu": torch.cuda.get_device_name(), "rows": []}
    for state in args.states.split(","):
        for n in [int(x) for x in args.ns.split(",")]:
            run(state, n, args, dev, kernel, head, out)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
