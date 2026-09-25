"""Audit vs force time per call. Usage: PYTHONPATH=. python scripts/fallback_overhead.py"""
import json

import numpy as np
import torch

from scripts import est_train
from scripts import kernels
from scripts import nbody_ic
from scripts.adaptive_force import AdaptiveForce

dev = torch.device("cuda")
kernel = est_train.make_kernel("analytic", dev)
kernels.current = kernel
head = est_train.load("checkpoints/est_analytic.pt", dev)
out = []
for n in (20000, 100000):
    for ic in ("flyby", "uniform"):
        pos, vel = nbody_ic.IC[ic](n, kernel, dev)
        for mode in ("geo_audit", "adaptive"):
            for k in (200, 1000, 3000):
                af = AdaptiveForce(kernel, head if mode == "adaptive" else None, mode=mode, audit_every=1, audit_k=k, target=1e9, device="cuda")
                for _ in range(8):
                    af(pos)
                    af.lam, af.theta = 1.0, 0.35
                st = af.stats[3:]
                row = {"n": n, "ic": ic, "mode": mode, "audit_k": k, "force_s": float(np.mean([s["time_s"] for s in st])),
                       "audit_s": float(np.mean([s["audit_time_s"] for s in st])), "cost": float(np.mean([s["cost"] for s in st]))}
                row["frac_at_every10"] = row["audit_s"] / 10 / (row["force_s"] + row["audit_s"] / 10)
                out.append(row)
                print(json.dumps(row), flush=True)
json.dump(out, open("results/fallback_overhead.json", "w"))
