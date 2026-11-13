"""Per-component cost of the 2D scatter-field control (ms_kp_pot_v_g128) and the 3D sparse-voxel model vs N, in a fixed box and in a box scaled to constant density.
Arms: bench-repro-run-model (exact bench.json path: ss.run_model, chunked all-pairs pair term, eager), bench-path-chunked-eager, cell-fast-io, adapt-fast-io, voxel-3d.
Results are written after every config to <out>/bench.json (key arm|scene|n) and finished keys are skipped on restart.
Usage: PYTHONPATH=. python scripts/scatter_field_2d_cost_blowup_bench.py --control kern/E_frozen.pt --voxel-ckpt ckpt/three_dimensional_sparse_voxel_scatter_field.pt --device cuda --out results/scatter_field_2d_cost_blowup
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling as ss
from scripts import scatter_scaling_ops as ops

SEED = 4738
STOP = {"flag": False}
FAST_IO = {"cell-fast-io": "kcache,fastio,cl,cell", "adapt-fast-io": "kcache,fastio,cl,adapt"}
PARTS_2D = ("scatter", "kernel_acc", "gather", "pp_acc", "far_acc")
PARTS_3D = ("structure", "deposit", "coarse_kernel", "gather_corners", "pair_acc")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def on_term(signum, frame):
    STOP["flag"] = True
    log("SIGTERM: finishing the current config, then exiting")


def plan(n):
    if n <= 10000:
        return dict(warm=3, steps=10, reps=5, cs=3)
    if n <= 100000:
        return dict(warm=1, steps=3, reps=3, cs=2)
    return dict(warm=1, steps=2, reps=2, cs=1)


def box(n, scene):
    s = math.sqrt(n / 1000.0) if scene == "constant-density-box" else 1.0
    return 8.0 * s, 64.0 * s


def box3(n, scene):
    return 8.0 * (n / 1000.0) ** (1.0 / 3.0) if scene == "constant-density-box" else 8.0


def save(res, path):
    json.dump(res, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


class Rec:
    def __init__(self, dev):
        self.cuda = dev.type == "cuda"
        self.ev = []

    def start(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def stop(self, name, s):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.ev.append((name, s, e))
        else:
            self.ev.append((name, s, time.perf_counter()))

    def totals(self):
        if self.cuda:
            torch.cuda.synchronize()
        out, cnt = {}, {}
        for name, s, e in self.ev:
            ms = s.elapsed_time(e) if self.cuda else (e - s) * 1000
            out[name] = out.get(name, 0.0) + ms
            cnt[name] = cnt.get(name, 0) + 1
        return out, cnt


def instrument(model, rec, parts, has_net):
    saved = {}
    for name in parts:
        if not hasattr(model, name):
            continue
        orig = getattr(model, name)
        saved[name] = model.__dict__.get(name, None)

        def wrap(*a, _o=orig, _n=name, **k):
            s = rec.start()
            r = _o(*a, **k)
            rec.stop(_n, s)
            return r

        setattr(model, name, wrap)
    hooks = []
    if has_net:
        stack = []
        hooks.append(model.net.register_forward_pre_hook(lambda m, i: stack.append(rec.start())))
        hooks.append(model.net.register_forward_hook(lambda m, i, o: rec.stop("net", stack.pop())))
    return saved, hooks


def restore(model, saved, hooks):
    for h in hooks:
        h.remove()
    for name, v in saved.items():
        if v is None:
            delattr(model, name)
        else:
            setattr(model, name, v)


def init_state(pos0, vel0, dev, model):
    n = pos0.shape[0]
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    ones = torch.ones(1, n, device=dev)
    return [pos, vel, model.init_field(1, dev), ones, ones]


def advance(model, st, k):
    pos, vel, field, mass, mask = st
    for _ in range(k):
        pos, vel, field, _ = model.step(pos, vel, mass, mask, field)
    return [pos, vel, field, mass, mask]


def occupancy(pos0, edge, dim, extent, n):
    p = torch.tensor(pos0, dtype=torch.float32)
    if dim == 2:
        c = int(math.ceil(extent / edge))
        i = ((p + extent / 2) / edge).floor().clamp(0, c - 1).long()
        key = i[:, 1] * c + i[:, 0]
    else:
        i = (p / edge).floor().long()
        i = i - i.min(0).values
        d = i.max(0).values + 1
        key = (i[:, 0] * d[1] + i[:, 1]) * d[2] + i[:, 2]
    counts = torch.unique(key, return_counts=True)[1]
    return {"max_cell_occupancy": int(counts.max()), "mean_occupied_cell": float(counts.float().mean()), "occupied_cells": int(counts.numel()),
            "frac_bodies_in_cells_over_32": float(counts[counts > 32].sum()) / n, "cells_over_32": int((counts > 32).sum())}


def run_config(model, pos0, vel0, n, dev, parts, has_net, pair_name):
    pl = plan(n)
    res = {"plan": pl}
    with torch.no_grad():
        st = init_state(pos0, vel0, dev, model)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        st = advance(model, st, pl["warm"])
        sync(dev)
        ts = []
        for _ in range(pl["reps"]):
            sync(dev)
            t = time.perf_counter()
            st = advance(model, st, pl["steps"])
            sync(dev)
            ts.append((time.perf_counter() - t) / pl["steps"] * 1000)
        res["ms_tick_reps"] = ts
        res["ms_tick_median"] = float(np.median(ts))
        res["ms_tick_min"], res["ms_tick_max"] = float(min(ts)), float(max(ts))
        res["ns_body_tick"] = res["ms_tick_median"] * 1e6 / n
        if dev.type == "cuda":
            res["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            res["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2 ** 30
        rec = Rec(dev)
        saved, hooks = instrument(model, rec, parts, has_net)
        sync(dev)
        t = time.perf_counter()
        st = advance(model, st, pl["cs"])
        tot, cnt = rec.totals()
        wall = (time.perf_counter() - t) * 1000
        restore(model, saved, hooks)
    nf = max(1, cnt.get(pair_name, pl["cs"]))
    comp = {k: v / nf for k, v in tot.items()}
    comp["total_wall"] = wall / nf
    comp["other"] = comp["total_wall"] - sum(v for k, v in comp.items() if k != "total_wall")
    res["component_ms_per_force"] = comp
    res["component_forces"] = nf
    if hasattr(model, "adapt_stats") and model.adapt_stats:
        res["adapt_stats"] = {"levels": model.adapt_stats.get("levels"), "fallback": model.adapt_stats.get("fallback")}
    if getattr(model, "cell_max_occ", None) is not None:
        res["cell_max_occ_runtime"] = int(model.cell_max_occ)
    if hasattr(model, "cell_overflow"):
        res["cell_overflow_cells_runtime"] = int(model.cell_overflow)
    return res


def repro_config(ctrl_path, n, dev):
    model = ss.load(ctrl_path, dev)
    sigma, extent = box(n, "fixed-box")
    ops.retarget(model, 128, extent)
    p2, v2 = ss.make_scene(n, sigma, 0.3, extent, SEED, 0)
    steps, reps = (10, 3) if n <= 10000 else (3, 2)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    ss.run_model(model, p2, v2, min(steps, 2), dev, probe=False)
    ts = []
    for _ in range(reps):
        sync(dev)
        t = time.perf_counter()
        ss.run_model(model, p2, v2, steps, dev, probe=False)
        sync(dev)
        ts.append((time.perf_counter() - t) / steps * 1000)
    res = {"ms_tick_reps": ts, "ms_tick_median": float(np.median(ts)), "plan": {"steps": steps, "reps": reps, "warm": "run_model(2)"}}
    if dev.type == "cuda":
        res["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    return res


def predicted_seconds(hist, n, pl):
    if not hist:
        return 0.0
    n1, m1 = hist[-1]
    ex = 2.0
    if len(hist) > 1:
        n0, m0 = hist[-2]
        ex = min(2.0, max(1.0, math.log(m1 / m0) / math.log(n1 / n0)))
    return m1 * (n / n1) ** ex * (2 + pl["steps"] * pl["reps"] + pl["cs"] + pl["warm"]) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="kern/E_frozen.pt")
    ap.add_argument("--voxel-ckpt", default="ckpt/three_dimensional_sparse_voxel_scatter_field.pt")
    ap.add_argument("--ns", default="1000,3000,10000,30000,100000,300000,1000000")
    ap.add_argument("--arms", default="bench-repro-run-model,bench-path-chunked-eager,cell-fast-io,adapt-fast-io,voxel-3d")
    ap.add_argument("--scenes", default="fixed-box,constant-density-box")
    ap.add_argument("--max-config-seconds", type=float, default=600.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/scatter_field_2d_cost_blowup")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    signal.signal(signal.SIGTERM, on_term)
    torch.manual_seed(SEED)
    path = f"{args.out}/bench.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    res["meta"] = {"seed": SEED, "args": vars(args), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    ns = [int(x) for x in args.ns.split(",")]
    arms, scenes = args.arms.split(","), args.scenes.split(",")
    v3 = None
    if "voxel-3d" in arms:
        from scripts import three_dimensional_sparse_voxel_scatter_field as v3
    hist = {}
    for key, v in res.items():
        if key != "meta" and "ms_tick_median" in v:
            a, s, n = key.split("|")
            hist.setdefault((a, s), []).append((int(n), v["ms_tick_median"]))
    for n in ns:
        for scene in scenes:
            for arm in arms:
                key = f"{arm}|{scene}|{n}"
                if key in res:
                    continue
                if arm == "bench-repro-run-model" and (scene != "fixed-box" or n not in (1000, 10000, 100000)):
                    continue
                if arm == "bench-path-chunked-eager" and scene == "constant-density-box" and n > 100000:
                    res[key] = {"skipped": "all-pairs cost is density-independent; fixed-box value applies"}
                    continue
                if STOP["flag"]:
                    save(res, path)
                    sys.exit(143)
                h = sorted(hist.get((arm, scene), []))
                est = predicted_seconds(h, n, plan(n))
                if est > args.max_config_seconds:
                    res[key] = {"skipped": "predicted over per-config limit", "predicted_seconds": est}
                    log(f"{key} skipped, predicted {est:.0f} s")
                    save(res, path)
                    continue
                sigma, extent = box(n, scene)
                try:
                    if arm == "bench-repro-run-model":
                        r = repro_config(args.control, n, dev)
                    elif arm == "voxel-3d":
                        a = box3(n, scene)
                        rng = np.random.default_rng([SEED, n, 0])
                        p3, vv3 = v3.plummer(n, a, 0.3, rng)
                        model = v3.sv.SparseVoxelScatterField(dt=v3.DT).to(dev)
                        model.load_state_dict(torch.load(args.voxel_ckpt, map_location=dev, weights_only=False)["model"])
                        model.eval()
                        r = run_config(model, p3, vv3, n, dev, PARTS_3D, False, "pair_acc")
                        r["pair_cell_occupancy"] = occupancy(p3, 2.0, 3, extent, n)
                        r["scene"] = {"plummer_scale": a}
                    else:
                        model = ss.load(args.control, dev, chunked=arm == "bench-path-chunked-eager")
                        ops.retarget(model, 128, extent)
                        if arm in FAST_IO:
                            sf.apply_opts(model, FAST_IO[arm])
                        p2, v2 = ss.make_scene(n, sigma, 0.3, extent, SEED, 0)
                        r = run_config(model, p2, v2, n, dev, PARTS_2D, True, "pp_acc")
                        r["pair_cell_occupancy"] = occupancy(p2, 2.0, 2, extent, n)
                        r["scene"] = {"sigma": sigma, "extent": extent, "grid": 128}
                except torch.cuda.OutOfMemoryError:
                    r = {"oom": True, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
                    torch.cuda.empty_cache()
                res[key] = r
                if "ms_tick_median" in r:
                    hist.setdefault((arm, scene), []).append((n, r["ms_tick_median"]))
                log(f"{key} {json.dumps({k: v for k, v in r.items() if k in ('ms_tick_median', 'peak_allocated_gb', 'oom')})}")
                save(res, path)
                model = None
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    save(res, path)
    log("DONE")


if __name__ == "__main__":
    main()
