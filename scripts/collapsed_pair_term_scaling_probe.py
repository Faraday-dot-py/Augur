"""Adaptive pair-term cost in a collapsed state (audit row 11; docs/debugging/director-feasibility-2026-11-07.md).
Times model.pp_acc (adaptive multilevel cell list, top-16) alone and one full model.force on a collapsed Plummer cloud (core scale 1, softening 0.5) at growing N, and records the
adaptive-path statistics. Cost per body per call and the log-log slope of pair-term time against N are the outputs. Cells are written after each N and skipped on rerun.
An OOM at some N is recorded as a result and stops the ladder.
Usage: PYTHONPATH=. python scripts/collapsed_pair_term_scaling_probe.py --ckpt kern/E_frozen.pt --out results/collapsed_pair_term_scaling_probe.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.pm_baseline_force_error import learned_acc, sample

NS = (100000, 300000, 1000000, 3000000)
SCALES = (("collapsed_core1", 0.25), ("reference_core4", 1.0))


def timed(fn, reps, dev):
    fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / reps


def add_slopes(res):
    for name, _ in SCALES:
        pts = sorted((c["n"], c["pair_term_s"]) for k, c in res["cells"].items() if k.startswith(name + "_n") and c["n"] <= 1000000)
        if len(pts) >= 2:
            x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            res[f"{name}_loglog_slope"] = float(np.polyfit(x, y, 1)[0])
            res[f"{name}_slope_last_two"] = float((y[-1] - y[-2]) / (x[-1] - x[-2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ns", default=",".join(str(n) for n in NS))
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    if a.ckpt != "random":
        model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    model.opt.add("adapt")
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"seed": a.seed, "cells": {}, "oom": None}
    rng = np.random.default_rng(a.seed)
    stop = False
    for n in [int(x) for x in a.ns.split(",")]:
        for name, scale in SCALES:
            key = f"{name}_n{n}"
            if key in res["cells"] or stop:
                continue
            try:
                pos = torch.tensor(sample("plummer", n, rng, scale), dtype=torch.float32, device=dev)[None]
                vel = torch.zeros_like(pos)
                mass = torch.ones(1, n, device=dev)
                mask = torch.ones(1, n, device=dev)
                reps = 3 if n <= 1000000 else 1
                with torch.no_grad():
                    t_pp = timed(lambda: model.pp_acc(pos, mass, mask, model.knn), reps, dev)
                    stats = json.loads(json.dumps(getattr(model, "adapt_stats", {}), default=str))
                    t_force = timed(lambda: learned_acc(model, pos, vel, mass, mask, 0), reps, dev)
                res["cells"][key] = {"n": n, "pair_term_s": t_pp, "full_force_s": t_force, "pair_ns_per_body": 1e9 * t_pp / n, "force_ns_per_body": 1e9 * t_force / n,
                                     "adapt_stats": stats, "peak_gb": float(torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else None}
                print(key, {k: res["cells"][key][k] for k in ("pair_term_s", "full_force_s", "pair_ns_per_body")}, flush=True)
            except torch.cuda.OutOfMemoryError:
                res["oom"] = key
                stop = True
                print(key, "OOM", flush=True)
            add_slopes(res)
            json.dump(res, open(a.out, "w"), indent=1)
            pos = vel = mass = mask = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    add_slopes(res)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", {k: v for k, v in res.items() if k.endswith("slope") or k.endswith("last_two")})


if __name__ == "__main__":
    main()
