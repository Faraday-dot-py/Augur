"""Memory ceiling of the est pipeline: increasing N under a per-process cap (default 120 GB of the 141 GB H200) until OOM.

Usage: PYTHONPATH=. python scripts/scaling_ceiling.py --impl opt --ns 2000000,4000000,8000000 --lam 1.0 --tag ceil_opt
"""
import argparse
import json
import time

import torch

from scripts import est_train
from scripts import kernels
from scripts import nbody_ic
from scripts import scaling_opt as so
from scripts.adaptive_force import AdaptiveForce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="opt", choices=["orig", "opt"])
    ap.add_argument("--ns", required=True)
    ap.add_argument("--ic", default="flyby")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--cap-gb", type=float, default=120.0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.cap_gb / total))
    kernel = kernels.analytic()
    kernels.current = kernel
    head = est_train.load("checkpoints/est_analytic.pt", dev)
    out = {"args": vars(args), "gpu": torch.cuda.get_device_name(), "total_gb": total, "rows": []}
    for n in [int(x) for x in args.ns.split(",")]:
        row = {"N": n}
        try:
            torch.manual_seed(4738)
            pos = nbody_ic.IC[args.ic](n, kernel, dev)[0]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if args.impl == "orig":
                af = AdaptiveForce(kernel, head, mode="est", device="cuda")
            else:
                af = so.OptForce(kernel, head, mode="est", opts=so.Opts(analytic_grad=True, compile=True), device="cuda")
            af.lam = args.lam
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            af(pos)
            torch.cuda.synchronize()
            row["cold_s"] = time.perf_counter() - t0
            row["peak_cold_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            t0 = time.perf_counter()
            af(pos)
            torch.cuda.synchronize()
            row["steady_s"] = time.perf_counter() - t0
            row["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            row["resident_gb"] = base / 2 ** 30
            row["cost"] = af.stats[-1]["cost"]
            del af, pos
        except torch.cuda.OutOfMemoryError as e:
            row["oom"] = True
            row["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            print(n, "OOM", flush=True)
            out["rows"].append(row)
            json.dump(out, open(f"results/scaling_{args.tag}.json", "w"))
            break
        print(row, flush=True)
        out["rows"].append(row)
        json.dump(out, open(f"results/scaling_{args.tag}.json", "w"))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
