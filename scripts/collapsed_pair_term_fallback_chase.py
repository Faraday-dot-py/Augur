"""Curiosity diagnosis: why does the adaptive pair term leave O(N) on a collapsed Plummer cloud between N=1e6 and 3e6
(docs/debugging/pair-term-scaling-fallback-chase.md). Per N and cloud: one warm-up call, REPS timed calls of model.pp_acc, then one profiled call that records
bodies per level, bodies over the 64-candidate cell cap per level, seconds per level and the chunked all-pairs fallback (bodies, chunks, seconds).
The profiled output is compared with the unprofiled output to confirm the instrumentation does not change numbers. Cells are written after each N and skipped on rerun.
Usage: PYTHONPATH=. python scripts/collapsed_pair_term_fallback_chase.py --ckpt kern/E_frozen.pt --out results/collapsed_pair_term_fallback_chase/chase.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.pm_baseline_force_error import sample

NS = (1000000, 1500000, 2000000, 3000000)
SCALES = (("collapsed_core1", 0.25), ("reference_core4", 1.0))
REPS = 3


def call(model, pos, mass, mask, dev):
    torch.cuda.synchronize()
    t = time.time()
    a = model.pp_acc(pos, mass, mask, model.knn)
    torch.cuda.synchronize()
    return a, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--ns", default=",".join(str(n) for n in NS))
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    if a.ckpt != "random":
        model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    model.opt.add("adapt")
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"seed": a.seed, "reps": REPS, "cells": {}}
    rng = np.random.default_rng(a.seed)
    for n in [int(x) for x in a.ns.split(",")]:
        for name, scale in SCALES:
            key = f"{name}_n{n}"
            pts = sample("plummer", n, rng, scale)
            if key in res["cells"]:
                continue
            pos = torch.tensor(pts, dtype=torch.float32, device=dev)[None]
            mass = torch.ones(1, n, device=dev)
            mask = torch.ones(1, n, device=dev)
            with torch.no_grad():
                model.adapt_profile = False
                ref, t_warm = call(model, pos, mass, mask, dev)
                ts = [call(model, pos, mass, mask, dev)[1] for _ in range(REPS)]
                model.adapt_profile = True
                prof_out, t_prof = call(model, pos, mass, mask, dev)
                model.adapt_profile = False
            st = json.loads(json.dumps(model.adapt_stats, default=str))
            res["cells"][key] = {"n": n, "warmup_s": t_warm, "steady_s": ts, "steady_mean_s": float(np.mean(ts)), "steady_ns_per_body": 1e9 * float(np.mean(ts)) / n,
                                 "profiled_call_s": t_prof, "max_abs_diff_profiled_vs_plain": float((prof_out - ref).abs().max()),
                                 "adapt_stats": st, "peak_gb": float(torch.cuda.max_memory_allocated() / 1e9)}
            print(key, {k: res["cells"][key][k] for k in ("warmup_s", "steady_s", "steady_ns_per_body", "max_abs_diff_profiled_vs_plain")}, "fallback", st["fallback"], flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
            pos = mass = mask = ref = prof_out = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", len(res["cells"]))


if __name__ == "__main__":
    main()
