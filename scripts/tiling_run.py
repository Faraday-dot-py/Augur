"""H6 tiling tests for the momentum-symmetric dual tree (scripts/tiling_emulator.py), one GPU.
--block accuracy: flyby frames 0/25/125/250 (N=100k), T in 2..32, x-strips / Morton blocks, designs A/B/C, geometric theta and estimator (fixed lam):
  force error vs exact and vs the global dual tree at the same setting, momentum, cost, imports per tile, halo counts, work balance.
--block scaling: imports per tile vs N, T for uniform / flyby / clumpy initial conditions (geometric theta, no exact reference).

Usage: PYTHONPATH=. python scripts/tiling_run.py --block accuracy --out results/tiling_accuracy.json
"""
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import est_train
from scripts import kernels
from scripts import nbody_ic
from scripts import tiling_emulator as te

FRAMES = [("t0", 0), ("t1000", 25), ("t5000", 125), ("t10000", 250)]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def agg(rows):
    own = np.array([r["own"] for r in rows], dtype=float)
    work = np.array([r["m2l"] + r["direct"] for r in rows], dtype=float)
    workw = np.array([4 * r["m2l"] + r["direct"] for r in rows], dtype=float)
    return {"own_min": int(own.min()), "own_max": int(own.max()), "count_imb": float(own.max() / own.mean()),
            "work_imb": float(work.max() / work.mean()), "workw_imb": float(workw.max() / workw.mean()),
            "cost_total": float(work.sum()), "m2l_total": int(sum(r["m2l"] for r in rows)),
            "imp_nodes_mean": float(np.mean([r["imp_nodes"] for r in rows])), "imp_nodes_max": int(max(r["imp_nodes"] for r in rows)),
            "imp_parts_mean": float(np.mean([r["imp_parts"] for r in rows])), "imp_parts_max": int(max(r["imp_parts"] for r in rows)),
            "imp_bytes_mean": float(np.mean([r["imp_bytes"] for r in rows])), "imp_bytes_max": int(max(r["imp_bytes"] for r in rows))}


def to_orig(F, a_sorted):
    a = torch.empty_like(a_sorted)
    a[F.order] = a_sorted
    return a


def eval_tiled(F, mode, cap, theta, head, lam, tol, theta_max):
    if mode == "geo":
        a_s, tiles = te.run_tiled(F, theta, cap)
        rows = te.summarize(F, tiles)
        return a_s, rows
    a_c, tiles_c = te.run_tiled(F, theta_max, cap)
    fs = de.node_scale(F, a_c)
    a_s, tiles = te.run_tiled(F, theta_max, cap, accept_fn=de.make_accept(F, head, lam, tol, fs))
    rows = te.summarize(F, tiles, est=True)
    for r, sc, sf in zip(rows, tiles_c, tiles):
        u = sc["touched"] | sf["touched"]
        r["imp_nodes_union"] = int((u & (sf["cnt_own"] < F.count)).sum())
        r["imp_nodes_coarse"] = te.import_counts(F, sc)[0]
        r["imp_parts_coarse"] = te.import_counts(F, sc)[1]
    return a_s, rows


def accuracy(args, dev, head):
    d = np.load(args.npz)
    out = {"args": vars(args), "rows": [], "frames": {}}
    for fname, fi in FRAMES:
        pos = torch.from_numpy(d["truth"][fi].copy()).to(dev)
        N = len(pos)
        S = torch.randperm(N, generator=torch.Generator().manual_seed(4738))[:args.eval_targets].to(dev)
        a_ex = ao.exact_accel(pos, S)
        ext = float((pos.max(0).values - pos.min(0).values).max())
        rad = float((pos - pos.median(0).values).norm(dim=1).median())
        cut = te.cutoff_errors(pos, S, args.radii)
        out["frames"][fname] = {"extent": ext, "median_radius": rad, "cutoff_rel_l2": {str(k): v for k, v in cut.items()}}
        print(fname, "extent", ext, "cutoff", cut, flush=True)
        tr = ao.Tree(pos, 18)
        fl = dt.Flat(tr)
        glob = {}
        a_c, _, _ = dt.dual_accel(tr, fl, args.theta_max, args.cap)
        fs = de.node_scale(fl, a_c)
        for mode in ("geo", "est"):
            if mode == "geo":
                a_s, m2l, dr = dt.dual_accel(tr, fl, args.theta, args.cap)
            else:
                a_s, m2l, dr = dt.dual_accel(tr, fl, args.theta_max, args.cap, accept_fn=de.make_accept(fl, head, args.lam, args.tol, fs))
            a = torch.empty_like(a_s)
            a[tr.order] = a_s
            glob[mode] = a
            g = {"frame": fname, "kind": "global", "mode": mode, "cost": (m2l + dr) / N, "net_ratio": ao.net_ratio(a), **de.metrics2(a[S], a_ex)}
            out["rows"].append(g)
            print(json.dumps(g), flush=True)
        rms = float(glob["geo"].norm(dim=1).pow(2).mean().sqrt())
        for T in args.Ts:
            for part in ("strips", "morton"):
                tid = te.partition(pos, T, part)
                halo = te.halo_counts(pos, tid, T, args.radii)
                out["rows"].append({"frame": fname, "kind": "halo", "T": T, "part": part,
                                    "halo_mean": {str(r): float(np.mean(v)) for r, v in halo.items()},
                                    "halo_max": {str(r): int(np.max(v)) for r, v in halo.items()}, "allgather_parts": N - N / T})
                for design in ("A", "B", "C"):
                    F = te.Forest(pos, tid, T, design)
                    for mode in ("geo", "est"):
                        t0 = sync()
                        a_s, rows = eval_tiled(F, mode, args.cap, args.theta, head, args.lam, args.tol, args.theta_max)
                        dtm = sync() - t0
                        a = to_orig(F, a_s)
                        ag = glob[mode]
                        r = {"frame": fname, "kind": "tiled", "T": T, "part": part, "design": design, "mode": mode, "time_s": dtm,
                             "net_ratio": ao.net_ratio(a), "diff_vs_global_rel": float((a - ag).norm() / ag.norm()),
                             "diff_vs_global_max_over_rms": float((a - ag).norm(dim=1).max() / rms),
                             "cost": sum(x["m2l"] + x["direct"] for x in rows) / N, "allgather_bytes": (N - N / T) * te.PART_BYTES,
                             **de.metrics2(a[S], a_ex), **agg(rows)}
                        if mode == "est":
                            r["imp_nodes_union_mean"] = float(np.mean([x["imp_nodes_union"] for x in rows]))
                            r["imp_nodes_coarse_mean"] = float(np.mean([x["imp_nodes_coarse"] for x in rows]))
                        out["rows"].append(r)
                        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
                    del F
                    torch.cuda.empty_cache()
        json.dump(out, open(args.out, "w"))


def scaling(args, dev):
    kernels.current = kernels.analytic()
    out = {"args": vars(args), "rows": []}
    for ic in args.ics:
        for N in args.Ns:
            pos = nbody_ic.IC[ic](N, kernels.current, dev)[0]
            for T in args.Ts:
                for design, part in (("C", "morton"), ("C", "strips"), ("B", "morton")):
                    tid = te.partition(pos, T, part)
                    F = te.Forest(pos, tid, T, design)
                    a_s, rows = eval_tiled(F, "geo", args.cap, args.theta, None, 0, 0, 0)
                    r = {"ic": ic, "N": N, "T": T, "part": part, "design": design, "cost": sum(x["m2l"] + x["direct"] for x in rows) / N,
                         "net_ratio": ao.net_ratio(to_orig(F, a_s)), "allgather_bytes": (N - N / T) * te.PART_BYTES, **agg(rows)}
                    out["rows"].append(r)
                    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
                    del F
                    torch.cuda.empty_cache()
            json.dump(out, open(args.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="accuracy")
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--est", default="checkpoints/est_analytic.pt")
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--Ns", type=int, nargs="+", default=[25000, 50000, 100000, 200000, 400000])
    ap.add_argument("--ics", nargs="+", default=["uniform", "flyby", "clumpy"])
    ap.add_argument("--radii", type=float, nargs="+", default=[10.0, 30.0, 100.0, 300.0, 1000.0])
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-max", type=float, default=1.2)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--eval-targets", type=int, default=20000)
    ap.add_argument("--out", default="results/tiling_accuracy.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernels.current = kernels.analytic()
    if args.block == "accuracy":
        accuracy(args, dev, est_train.load(args.est, dev))
    else:
        scaling(args, dev)


if __name__ == "__main__":
    main()
