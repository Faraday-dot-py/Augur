"""genic task 3: particle-count generalization of the flyby-trained (N=100k) estimator, N in 10k..300k, flyby and uniform (analytic kernel).
Exact reference on 10k random targets. Fixed-tol cost/error, audited (target 0.01, static state, repeated calls) cost/error, calibration.

Usage: PYTHONPATH=. python scripts/genic_scaling.py --out results/genic_scaling.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import est_train
from scripts import genic_lib as gl
from scripts import kernels
from scripts import nbody_ic
from scripts.adaptive_force import AdaptiveForce


def audited(pos, mode, head, S, a_ex, dev, calls=60):
    af = AdaptiveForce(kernels.current, head, mode=mode, target=0.01, audit_every=1, audit_k=1000, device=str(dev))
    for _ in range(calls):
        a = af(pos)
    st = af.stats
    return {"mode": mode, "cost_last10": float(np.mean([s["cost"] for s in st[-10:]])), "cost_first": st[0]["cost"], "lam": af.lam, "theta": af.theta,
            "final_now": af.now, "events": af.events, "audit_last10": float(np.mean([s["audit"] for s in st[-10:]])),
            "time_last10": float(np.mean([s["time_s"] for s in st[-10:]])), **de.metrics2(a[S], a_ex)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[10000, 30000, 100000, 300000])
    ap.add_argument("--ics", default="flyby,uniform")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    k = kernels.analytic()
    kernels.current = k
    head = est_train.load("checkpoints/est_analytic.pt", dev)
    gen = torch.Generator(device=dev).manual_seed(4738)
    out = {}
    for ic in a.ics.split(","):
        for n in a.ns:
            key = f"{ic}_{n}"
            pos = nbody_ic.IC[ic](n, k, dev)[0]
            S = gl.make_targets(n, 10000, dev)
            r = gl.sweep(pos, {"flyby": head}, S, tols=[3e-4, 1e-3, 3e-3, 1e-2], bh=False)
            r["calib"] = gl.calib(pos, head, gen)
            a_ex = ao.exact_accel(pos, S)
            r["audited"] = [audited(pos, m, head, S, a_ex, dev) for m in ("adaptive", "geo_audit")]
            out[key] = r
            print(key, json.dumps(r["calib"]), flush=True)
            for x in r["audited"]:
                print(key, "audited", json.dumps({q: x[q] for q in ("mode", "cost_last10", "rel_l2", "abs_p99", "lam", "theta", "final_now")}), flush=True)
            json.dump(out, open(a.out, "w"))
            del pos
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
