"""Verify and time the scaling_opt.py speedups against the original ALF force call.

Usage: PYTHONPATH=. python scripts/scaling_opt_bench.py --ns 100000,1000000 --tag opt
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import est_train
from scripts import kernels
from scripts import nbody_ic
from scripts import scaling_opt as so
from scripts import scaling_timed as st
from scripts.adaptive_force import AdaptiveForce


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def med(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = sync()
        out = fn()
        ts.append(sync() - t0)
    return float(np.median(ts)), out


def diff(a, b):
    d = (a - b).norm(dim=1)
    return {"max_abs": float(d.max()), "rel_max": float((d / b.norm(dim=1)).max()), "rel_l2": float(d.norm() / b.norm())}


def peak(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="100000,1000000")
    ap.add_argument("--state", default="flyby")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    kernel = kernels.analytic()
    kernels.current = kernel
    torch.manual_seed(4738)
    head = est_train.load("checkpoints/est_analytic.pt", dev)
    out = {"args": vars(args), "gpu": torch.cuda.get_device_name(), "rows": []}
    for n in [int(x) for x in args.ns.split(",")]:
        pos = nbody_ic.IC[args.state](n, kernel, dev)[0]
        n = pos.shape[0]
        r = {"N": n}
        print("== N", n, flush=True)
        tr0 = ao.Tree(pos, 18)
        fl0 = dt.Flat(tr0)
        tr1 = so.OTree(pos, 18)
        fl1 = so.oflat(tr1, 8)
        r["tree_check"] = {"count_equal": bool(torch.equal(fl0.count, fl1.count)), "child_equal": bool(torch.equal(fl0.child, fl1.child)),
                           "pnode_equal": bool(torch.equal(fl0.pnode, fl1.pnode)), "start_equal": bool(torch.equal(fl0.start, fl1.start)),
                           "com_max_abs": float((fl0.com - fl1.com).abs().max()), "q_max_rel": float(((fl0.q - fl1.q).abs() / (fl0.q.abs() + 1e-12)).max()),
                           "nodes": len(fl0.count)}
        print("tree check", r["tree_check"], flush=True)
        del tr0, fl0, tr1, fl1
        af0 = AdaptiveForce(kernel, head, mode="est", device="cuda")
        af0.lam = args.lam
        af0(pos)
        prev = af0.prev_a.clone()
        af0.prev_a = prev
        a0 = af0(pos)
        af0.prev_a = prev
        a0b = af0(pos)
        r["orig_self_diff"] = diff(a0b, a0)
        r["cost_orig"] = af0.stats[-1]["cost"]

        def call0():
            af0.prev_a = prev
            return af0(pos)
        t0, _ = med(call0, args.reps)
        r["orig_s"] = t0
        r["orig_peak_gb"] = peak(call0)
        print("orig", t0, r["orig_self_diff"], flush=True)
        ladder = [("sync_light", so.Opts(fast_tree=False)), ("fast_tree", so.Opts()), ("analytic_grad", so.Opts(analytic_grad=True)),
                  ("compile", so.Opts(analytic_grad=True, compile=True)), ("compile_generic_grad", so.Opts(compile=True))]
        r["ladder"] = {}
        for name, o in ladder:
            try:
                of = so.OptForce(kernel, head, mode="est", opts=o, device="cuda")
                of.lam = args.lam

                def call1():
                    of.prev_a = prev
                    return of(pos)
                a1 = call1()
                t1, _ = med(call1, args.reps)
                row = {"time_s": t1, "speedup": t0 / t1, "cost": of.stats[-1]["cost"], **diff(a1, a0), "peak_gb": peak(call1)}
                T = st.Tm()
                of.T = T
                for _ in range(2):
                    T.d.clear()
                    call1()
                row["breakdown_s"] = dict(T.d)
                of.T = None
                r["ladder"][name] = row
                print(name, row, flush=True)
            except Exception as e:
                r["ladder"][name] = {"error": repr(e)[:400]}
                print(name, "FAILED", repr(e)[:400], flush=True)
        T = st.Tm()
        af0.prev_a = prev
        tim = []
        for _ in range(3):
            T = st.Tm()
            af0.prev_a = prev
            st.timed_call(pos, af0, T, "est")
            tim.append(dict(T.d))
        r["orig_breakdown_s"] = {k: float(np.median([t[k] for t in tim])) for k in tim[0]}
        print("orig breakdown", r["orig_breakdown_s"], flush=True)
        g0 = AdaptiveForce(kernel, head, mode="geo", device="cuda")
        g0(pos)
        tg0, ag0 = med(lambda: g0(pos), args.reps)
        g1 = so.OptForce(kernel, head, mode="geo", opts=so.Opts(analytic_grad=True, compile=True), device="cuda")
        g1(pos)
        tg1, ag1 = med(lambda: g1(pos), args.reps)
        r["geo"] = {"orig_s": tg0, "opt_s": tg1, "speedup": tg0 / tg1, **diff(ag1, ag0), "cost_orig": g0.stats[-1]["cost"], "cost_opt": g1.stats[-1]["cost"]}
        print("geo", r["geo"], flush=True)
        gen0 = torch.Generator(device=dev).manual_seed(4738)
        gen1 = torch.Generator(device=dev).manual_seed(4738)
        e0 = de.audit_e2e(pos, prev, 1000, gen0)
        e1 = so.audit_fast(pos, prev, 1000, gen1)
        ta0, _ = med(lambda: de.audit_e2e(pos, prev, 1000, gen0), args.reps)
        ta1, _ = med(lambda: so.audit_fast(pos, prev, 1000, gen1), args.reps)
        r["audit"] = {"orig_s": ta0, "fast_s": ta1, "speedup": ta0 / ta1, "err_orig": e0, "err_fast": e1}
        print("audit", r["audit"], flush=True)
        if n <= 100000:
            idx = torch.arange(n, device=dev)
            te0, x0 = med(lambda: ao.exact_accel(pos, idx, 1024), min(args.reps, 3))
            te1, x1 = med(lambda: so.exact_fast(pos, idx, 1024), min(args.reps, 3))
            r["exact"] = {"orig_s": te0, "fast_s": te1, "speedup": te0 / te1, **diff(x1, x0)}
            print("exact", r["exact"], flush=True)
        out["rows"].append(r)
        json.dump(out, open(f"results/scaling_{args.tag}.json", "w"))
        del pos, af0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
