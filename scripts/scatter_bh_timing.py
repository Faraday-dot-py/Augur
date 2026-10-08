"""Timing-only rollout of the scatter-field model on the black-hole scene at large N (no reference, no accuracy).

Same scene as scatter_bh_large.py (Gaussian cluster sigma 8, vfac 0.3, c 10, relativistic kick-drift-kick, dt 0.1) but initial conditions are generated on the GPU
(seeded) so nothing large lives on the host. --pair cell: scatter-opt cell-list pair term; --pair none: mesh only. Writes results/scatter_bh_timing_<N>_<pair>.json.

Usage: PYTHONPATH=. python scripts/scatter_bh_timing.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 1000000 --pair none
"""
import argparse
import json
import math
import statistics
import time

import torch

from scripts import scatter_field as sf
from scripts import scatter_bh as sb
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

CELL_S_PER_CAND = 1.7e-10


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--pair", choices=["cell", "none"], default="cell")
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--max-proj-s", type=float, default=3600.0)
    ap.add_argument("--skip-est-s", type=float, default=900.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    n, c, dt = args.n, args.c, args.dt
    tag = args.tag or f"scatter_bh_timing_{n}_{args.pair}"
    res = {"n": n, "pair": args.pair, "steps_requested": args.steps, "sigma": args.sigma, "vfac": args.vfac, "c": c, "dt": dt, "seed": args.seed,
           "dtype": "float32", "ckpt": args.ckpt, "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"}

    def write():
        json.dump(res, open(f"results/{tag}.json", "w"), indent=1)

    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", dt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    if args.pair == "cell":
        sf.apply_opt(model, "cell")
    else:
        ops.no_pp(model)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        g = torch.Generator(device=dev)
        g.manual_seed(args.seed)
        pos = (torch.randn(1, n, 2, device=dev, generator=g) * args.sigma).clamp_(-29.0, 29.0)
        vel = torch.randn(1, n, 2, device=dev, generator=g) * (args.vfac * math.sqrt(n / (4 * args.sigma)))
        vel -= vel.mean(1, keepdim=True)
        mass = torch.ones(1, n, device=dev)
        mask = torch.ones(1, n, device=dev)
        vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
        mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
        del vel, vmag
        field = model.init_field(1, dev)[0]

        if args.pair == "cell":
            counts = model._cell_keys(pos, mask)[7]
            m0 = int(counts[:-1].max())
            est = n * 9 * m0 * CELL_S_PER_CAND
            res.update(max_cell_occupancy_t0=m0, est_tick_s_from_candidates=est)
            if est > args.skip_est_s:
                res.update(skipped=True, reason=f"estimated {est:.0f} s/tick at tick 0 (N*9*M = {n * 9 * m0:.3e} candidates) exceeds {args.skip_est_s:.0f} s; cell search not run")
                print(res["reason"], flush=True)
                write()
                return

        ms, vmax_c, out_frac, out_ticks = [], [], [], []
        with torch.no_grad():
            sync(dev)
            t0 = time.perf_counter()
            a, field = model.force(pos, sb.speed(mom, c), mass, mask, field)
            sync(dev)
            res["init_force_ms"] = (time.perf_counter() - t0) * 1e3
            print(f"init force {res['init_force_ms']:.1f} ms", flush=True)
            t_all = time.perf_counter()
            for t in range(args.steps):
                sync(dev)
                t0 = time.perf_counter()
                mom = mom + 0.5 * dt * a
                pos = pos + dt * sb.speed(mom, c)
                a, field = model.force(pos, sb.speed(mom, c), mass, mask, field)
                mom = mom + 0.5 * dt * a
                sync(dev)
                ms.append((time.perf_counter() - t0) * 1e3)
                if t % 5 == 0 or t == args.steps - 1:
                    out_ticks.append(t + 1)
                    out_frac.append(float((pos.abs() > model.extent / 2).any(-1).float().mean()))
                    vmax_c.append(float(sb.speed(mom, c).norm(dim=-1).max() / c))
                    print(f"tick {t + 1} {ms[-1]:.1f} ms frac outside grid {out_frac[-1]:.3f} vmax/c {vmax_c[-1]:.4f}", flush=True)
                res.update(ms_per_tick=ms, steps_done=len(ms), diag_ticks=out_ticks, frac_outside_grid=out_frac, vmax_over_c=vmax_c)
                if args.pair == "cell":
                    res["max_cell_occupancy_last"] = int(model.cell_max_occ)
                proj = max(statistics.mean(ms), ms[-1]) * args.steps / 1e3
                if len(ms) < args.steps and proj > args.max_proj_s and (len(ms) >= 5 or time.perf_counter() - t_all > 1500):
                    res.update(truncated=True, projected_total_s=proj, reason=f"stopped after {len(ms)} ticks: projected {proj:.0f} s for {args.steps} ticks exceeds {args.max_proj_s:.0f} s")
                    print(res["reason"], flush=True)
                    break
    except torch.cuda.OutOfMemoryError as e:
        res.update(oom=True, oom_message=str(e).splitlines()[0])
        print("OOM", res["oom_message"], flush=True)
    if dev.type == "cuda":
        res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    ms = res.get("ms_per_tick", [])
    if ms:
        res.update(first_tick_ms=ms[0], median_ms=statistics.median(ms), total_ms=sum(ms), ns_per_body_per_tick=statistics.median(ms) * 1e6 / n)
        print(f"n {n} pair {args.pair}: first {ms[0]:.1f} ms median {res['median_ms']:.1f} ms total {sum(ms) / 1e3:.1f} s peak {res.get('peak_mem_gb', 0):.2f} GB "
              f"{res['ns_per_body_per_tick']:.1f} ns/body/tick", flush=True)
    write()


if __name__ == "__main__":
    main()
