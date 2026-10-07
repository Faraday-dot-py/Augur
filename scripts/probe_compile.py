"""torch.compile / Triton probe for the scatter field on Polaris: simple fn, then ScatterField._force (N=100/1000) vs eager and graph stack.

Usage: PYTHONPATH=. python scripts/probe_compile.py --ckpt <ckpt> [--mode default|max-autotune-no-cudagraphs]
"""
import argparse
import time
import traceback

import torch

from scripts import scatter_field as sf
from scripts import scatter_opt_bench as sb
from scripts import train_scatter_field as tsf


def timeit(fn, reps=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--ns", default="100,1000")
    args = ap.parse_args()
    dev = torch.device("cuda")
    print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
    try:
        import triton
        print("triton", triton.__version__, flush=True)
        x = torch.randn(1024, device=dev)
        f = torch.compile(lambda a: torch.sin(a) * a + 1.0)
        print("PROBE_SIMPLE ok", (f(x) - (torch.sin(x) * x + 1.0)).abs().max().item(), flush=True)
    except Exception:
        print("PROBE_SIMPLE FAILED", flush=True)
        traceback.print_exc()
        return
    m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    m.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    ref = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    ref.load_state_dict(m.state_dict())
    sf.apply_opts(m, "kcache,fastio,cl")
    sf.apply_opts(ref, "default")
    cm = torch.compile(m._force, mode=args.mode)
    for n in [int(x) for x in args.ns.split(",")]:
        pos, vel, mass, mask = sb.scene(n, 1, dev)
        field = m.init_field(1, dev)[0]
        with torch.no_grad():
            try:
                t0 = time.perf_counter()
                a_c = cm(pos, vel, mass, mask, field)[0]
                print(f"N {n}: compile+first call {time.perf_counter() - t0:.1f}s", flush=True)
                a_r = ref.force(pos, vel, mass, mask, field)[0]
                print(f"N {n}: max|a_compiled-a_graphstack| {(a_c - a_r).abs().max().item():.3e} max|a| {a_r.abs().max().item():.3e}", flush=True)
                ms_e = timeit(lambda: m._force(pos, vel, mass, mask, field))
                ms_g = timeit(lambda: ref.force(pos, vel, mass, mask, field))
                ms_c = timeit(lambda: cm(pos, vel, mass, mask, field))
                print(f"N {n}: force ms eager(kcache,fastio,cl) {ms_e:.3f} graph {ms_g:.3f} compiled[{args.mode}] {ms_c:.3f}", flush=True)
            except Exception:
                print(f"N {n}: COMPILED FORCE FAILED", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()
