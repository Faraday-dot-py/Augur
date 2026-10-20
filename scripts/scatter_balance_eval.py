"""Load balance across G=128 tiles on cold-collapse scenes (docs/debugging/scatter-tiling.md, round 2 section 3).

Domain 128 (extent 128, h .5), 4x4 uniform cores of 32 (halo 16) vs quadtree refinement by body count (`split_cores`), `adapt` pair term.
For states taken from a single-grid (G=256) rollout: per-tile own/ghost counts, per-tile solo force time, LPT packing onto 16 workers
(max/mean worker time), total work, batched time, and force error vs the single grid by seam class. Then 20-tick rollouts (single, uniform,
split with retile every 5 ticks) vs exact fp64.
Usage: PYTHONPATH=. python scripts/scatter_balance_eval.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/scatter_balance.json
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling_cores import CoreTiled, lpt_assign, split_cores, uniform_cores
from scripts.scatter_tiling_eval import DT, EPS, SUB, class_masks, load, rel_by_class

L, WORKERS = 128.0, 16
TICKS = (0, 3, 6, 10, 20, 40)


def tmed(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def schemes(pos, n):
    out = {"uniform": uniform_cores()}
    for kk in (1, 2, 4, 8):
        out[f"split_k{kk}"] = split_cores(uniform_cores(), pos, n / (kk * WORKERS))
    return out


@torch.no_grad()
def eval_state(model, pos, vel, mass, f1, n):
    msk = torch.ones(1, n, device=pos.device)
    model.grid, model.extent, model.h = 256, L, 0.5
    a_s = model.force(pos[None], vel[None], mass[None], msk, f1)[0][0]
    res = {}
    for name, cores in schemes(pos, n).items():
        T = CoreTiled(model, cores, far=True)
        fld = T.slice_fields(f1[0])
        a_t, _ = T.force(pos, vel, mass, fld)
        stats = dict(T.stats)
        ds, dout = T.seam_dist(pos)
        rel = rel_by_class(a_t, a_s, class_masks(ds, dout))
        ms = [tmed(lambda k=k: T.force(pos, vel, mass, fld, tiles=[k])) for k in range(T.T)]
        batched = tmed(lambda: T.force(pos, vel, mass, fld), reps=3)
        T.far = False
        batched_nofar = tmed(lambda: T.force(pos, vel, mass, fld), reps=3)
        T.far = True
        bins = [list(range(T.T))] if False else lpt_assign(ms, WORKERS)
        wt = [sum(ms[k] for k in b) for b in bins]
        own = stats["own"]
        res[name] = {"tiles": T.T, "own": own, "ghost": stats["ghost"], "tile_ms": ms, "worker_ms": wt,
                     "worker_max_over_mean": max(wt) / (sum(wt) / WORKERS), "tile_max_over_mean": max(ms) / (sum(ms) / T.T),
                     "own_max_over_mean": max(own) / (sum(own) / T.T), "makespan_ms": max(wt), "total_work_ms": sum(ms),
                     "batched_ms": batched, "batched_nofar_ms": batched_nofar, "force_vs_single": rel,
                     "cores": [list(c) for c in cores]}
        s = rel["seam<8"]["rel"]
        i = rel["interior"]["rel"]
        res[name]["seam_over_interior"] = (s / max(i, 1e-3)) if s is not None and i is not None else None
        print(name, {k: v for k, v in res[name].items() if k in ("tiles", "worker_max_over_mean", "tile_max_over_mean", "own_max_over_mean", "makespan_ms",
                                                                  "total_work_ms", "batched_ms", "seam_over_interior")}, flush=True)
    return res


@torch.no_grad()
def phase(model, dev, out, ns, seeds):
    res = {}
    for n in ns:
        for seed in seeds:
            sigma = 16.0
            p0, v0 = make_scene(n, sigma, 0.3, L, seed, 0)
            pos = torch.tensor(p0, dtype=torch.float32, device=dev)
            vel = torch.tensor(v0, dtype=torch.float32, device=dev)
            mass = torch.ones(n, device=dev)
            msk = torch.ones(1, n, device=dev)
            model.grid, model.extent, model.h = 256, L, 0.5
            st = (model.init_field(1, dev)[0], None)
            p, v = pos[None], vel[None]
            Ps = [p[0]]
            states = {}
            for t in range(max(TICKS) + 1):
                if t in TICKS:
                    states[t] = (p[0].clone(), v[0].clone(), st[0].clone())
                p, v, st, _ = model.step(p, v, mass[None], msk, st)
                Ps.append(p[0])
            key = f"N{n}_seed{seed}"
            res[key] = {"states": {}}
            for t in TICKS:
                pt, vt, ft = states[t]
                print(f"== {key} tick {t}", flush=True)
                res[key]["states"][str(t)] = eval_state(model, pt, vt, mass, ft, n)
                json.dump(res, open(out, "w"))
            if n <= 20000:
                Pt, _ = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), 20, DT, SUB, EPS)
                Ps = torch.stack(Ps).double()
                roll = {"single": float((Ps[20] - Pt[20]).norm(dim=-1).mean()), "const_vel": float((Pt[0] + torch.tensor(v0, device=dev) * DT * 20 - Pt[20]).norm(dim=-1).mean())}
                for name, k in (("uniform", None), ("split_k4", 4)):
                    T = CoreTiled(model, uniform_cores(), far=True)
                    ts = (T.init_field(dev), None)
                    p, v = pos, vel
                    for tick in range(20):
                        if k is not None and tick % 5 == 0:
                            ts = (T.retile(split_cores(uniform_cores(), p, n / (k * WORKERS)), ts[0]), ts[1])
                        p, v, ts = T.step(p, v, mass, ts)
                    roll[name] = float((p.double() - Pt[20]).norm(dim=-1).mean())
                    roll[name + "_diff_single"] = float((p.double() - Ps[20]).norm(dim=-1).mean())
                    roll[name + "_dP"] = float((v.sum(0).norm() - vel.sum(0).norm()).abs())
                res[key]["roll20"] = roll
                print(key, "roll20", roll, flush=True)
                json.dump(res, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="20000,60000")
    ap.add_argument("--seeds", default="4738,9000")
    ap.add_argument("--adapt-min", type=int, default=6000)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    model = load(a.ckpt, dev)
    sf.apply_opt(model, "adapt")
    model.adapt_min = a.adapt_min
    phase(model, dev, a.out, [int(x) for x in a.ns.split(",")], [int(x) for x in a.seeds.split(",")])


if __name__ == "__main__":
    main()
