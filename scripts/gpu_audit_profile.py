import argparse
import json
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from scripts import train_scatter_field as T

START, STOP = 30, 60
state = {"n": 0, "prof": None}


def union_ms(intervals):
    intervals.sort()
    tot = 0.0
    cur_s, cur_e = None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot / 1000.0


def analyse(prof, out):
    kern, cpu_sync = [], 0.0
    n_k = 0
    ksum = 0.0
    names = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kern.append((e.time_range.start, e.time_range.end))
            n_k += 1
            ksum += e.time_range.elapsed_us()
            names[e.name] = names.get(e.name, 0.0) + e.time_range.elapsed_us()
        elif e.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "aten::item", "aten::_local_scalar_dense"):
            cpu_sync += e.time_range.elapsed_us()
    t0 = min(s for s, _ in kern)
    t1 = max(e for _, e in kern)
    wall = (t1 - t0) / 1000.0
    busy = union_ms(kern)
    top = sorted(names.items(), key=lambda x: -x[1])[:8]
    res = {"window_iters": STOP - START, "wall_ms": wall, "gpu_busy_ms": busy, "gpu_busy_frac": busy / wall, "n_kernels": n_k,
           "mean_kernel_us": ksum / n_k, "kernels_per_iter": n_k / (STOP - START), "wall_ms_per_iter": wall / (STOP - START),
           "sync_ms_cpu": cpu_sync / 1000.0, "top_kernels_ms": [(k[:70], v / 1000.0) for k, v in top]}
    print("PROFILE " + json.dumps(res), flush=True)
    json.dump(res, open(out, "w"))


def main():
    out = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    orig_parse = argparse.ArgumentParser.parse_args

    def parse(self, *a, **k):
        args = orig_parse(self, *a, **k)

        def hook(idx):
            state["n"] += 1
            if state["n"] == START:
                torch.cuda.synchronize()
                state["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                state["prof"].__enter__()
            elif state["n"] == STOP:
                torch.cuda.synchronize()
                state["prof"].__exit__(None, None, None)
                analyse(state["prof"], out)
                sys.exit(0)

        args.batch_hook = hook
        return args

    argparse.ArgumentParser.parse_args = parse
    T.main()


main()
