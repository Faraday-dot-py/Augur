"""Stage 1 of three-dimensional-sparse-voxel-scatter-field: train model/sparse_voxel_scatter_field.py on 3D scenes with exact 3D truth, score on 3D Plummer spheres at N=1000 against the
frozen 2D the scatter-field control (ms_kp_pot_v_g128) (ratio of err@20 to constant-velocity err@20, same seeds, same N), and time both from 1e3 to 1e5 bodies.
Modes: train (resumable, SIGTERM-safe, periodic full-state checkpoints; --stop-at N saves and exits 0 at iteration N without train_done.json), score (--ckpt-file, --out for a slice), bench, all.
Usage: PYTHONPATH=. python scripts/three_dimensional_sparse_voxel_scatter_field.py --mode all --control kern/E_frozen.pt --iters 10000 --device cuda --out results/three_dimensional_sparse_voxel_scatter_field
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import time

import numpy as np
import torch

from model import sparse_voxel_scatter_field as sv
from scripts import scatter_field as sf
from scripts import scatter_scaling as ss
from scripts import scatter_scaling_ops as ops
from scripts import stage1_common as common

DT, EPS, SUB = 0.1, 0.5, 4
HELD_OUT = (9100, 9200, 9300)
VAL_SEED = 9400
TICKS = (5, 10, 20, 50, 100)
BENCH_NS = (1000, 10000, 100000)
STOP = {"flag": False}


def log(s):
    print(s, flush=True)


def on_term(signum, frame):
    STOP["flag"] = True


def plummer(n, a, q, rng):
    """Unit-mass Plummer sphere, G=1, total mass n, velocities from the exact Plummer distribution function scaled by sqrt(q) (q=1 virial). Returns (n,3), (n,3), centred, zero net momentum."""
    r = a / np.sqrt(rng.uniform(1e-4, 1.0, n) ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(n, 3))
    pos = r[:, None] * d / np.linalg.norm(d, axis=1, keepdims=True)
    vesc = np.sqrt(2.0 * n / np.sqrt(r ** 2 + a ** 2))
    x = np.empty(0)
    while x.shape[0] < n:
        xi, yi = rng.uniform(0, 1, 4 * n), rng.uniform(0, 0.1, 4 * n)
        x = np.concatenate([x, xi[yi < xi ** 2 * (1 - xi ** 2) ** 3.5]])
    x = x[:n]
    d = rng.normal(size=(n, 3))
    vel = (x * vesc * np.sqrt(q))[:, None] * d / np.linalg.norm(d, axis=1, keepdims=True)
    return pos - pos.mean(0), vel - vel.mean(0)


def cold_ball(n, radius, vfac, rng):
    d = rng.normal(size=(n, 3))
    pos = radius * rng.uniform(0, 1, n)[:, None] ** (1 / 3) * d / np.linalg.norm(d, axis=1, keepdims=True)
    vel = rng.normal(0, vfac * np.sqrt(n / radius), (n, 3))
    return pos - pos.mean(0), vel - vel.mean(0)


def train_scenes(num, steps, seed, dev):
    """the scatter-field control's mix in 3D: n in [10, 200]; half Plummer spheres (a in [2, 6], q in [0.2, 1]), half cold balls (radius 5-10, vfac 0.1-0.6), exact truth, unit mass."""
    rng = np.random.default_rng(seed)
    Nmax = 200
    P = torch.zeros(num, steps + 1, Nmax, 3, dtype=torch.float64)
    V = torch.zeros_like(P)
    M = torch.zeros(num, Nmax, dtype=torch.float64)
    mask = torch.zeros_like(M)
    for s in range(num):
        n = int(rng.integers(10, Nmax + 1))
        if s % 2 == 0:
            p0, v0 = plummer(n, rng.uniform(2, 6), rng.uniform(0.2, 1.0), rng)
        else:
            p0, v0 = cold_ball(n, rng.uniform(5, 10), rng.uniform(0.1, 0.6), rng)
        Pt, Vt = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), steps, DT, SUB, EPS)
        P[s, :, :n], V[s, :, :n], M[s, :n], mask[s, :n] = Pt.cpu(), Vt.cpu(), 1.0, 1.0
    return P.to(dev), V.to(dev), M.to(dev), mask.to(dev)


def val_err(model, val, dev):
    P, V, M, mask = val
    with torch.no_grad():
        Pm, _ = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), P.shape[1] - 1)
    e = ((Pm.double() - P).norm(dim=-1) * mask[:, None]).sum(2) / mask.sum(1)[:, None]
    return float(e[:, 20].mean())


def build(dev):
    return sv.SparseVoxelScatterField(dt=DT).to(dev)


def train(args, dev):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = ckpt_path(args)
    if args.stop_at > args.iters:
        raise SystemExit(f"--stop-at {args.stop_at} exceeds --iters {args.iters}")
    args.exact_resume = True
    if args.stop_at == args.iters:
        args.stop_at = 0
    if args.stop_at and os.path.exists(ckpt):
        have = torch.load(ckpt, map_location="cpu", weights_only=False)["it"]
        if have >= args.stop_at:
            raise SystemExit(f"checkpoint already at it {have}, not below --stop-at {args.stop_at}")
    model = build(dev)
    tensors = train_scenes(args.train, args.steps, args.seed, dev)
    val = train_scenes(args.val_scenes, 20, VAL_SEED, dev)
    n_it, secs = common.train_with_curve(model, tensors, lambda m: val_err(m, val, dev), args, ckpt, args.out, lambda: STOP["flag"], log)
    if args.stop_at and n_it == args.stop_at:
        if torch.load(ckpt, map_location="cpu", weights_only=False)["it"] != args.stop_at:
            raise SystemExit(f"checkpoint it is not {args.stop_at}, refusing to copy a stale slice")
        shutil.copyfile(ckpt, f"{args.ckpt_dir}/three_dimensional_sparse_voxel_scatter_field-at-{args.stop_at}-its.pt")
        log(f"STOP-AT {args.stop_at}: checkpoint saved and copied to {args.ckpt_dir}/three_dimensional_sparse_voxel_scatter_field-at-{args.stop_at}-its.pt, train_done.json not written")
        if not STOP["flag"]:
            raise SystemExit(0)
    if STOP["flag"]:
        log("SIGTERM: checkpoint saved, exiting for requeue")
        raise SystemExit(143)
    json.dump({"train_iters": n_it, "train_seconds": secs, "args": vars(args)}, open(f"{args.out}/train_done.json", "w"))


def ckpt_path(args):
    return args.ckpt_file or f"{args.ckpt_dir}/three_dimensional_sparse_voxel_scatter_field.pt"


def load_candidate(args, dev):
    model = build(dev)
    st = torch.load(ckpt_path(args), map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    args.ckpt_it = st["it"]
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def run3(model, pos0, vel0, steps, dev):
    P = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    V = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    M = torch.ones(1, pos0.shape[0], device=dev)
    with torch.no_grad():
        Pm, _ = sf.rollout(model, P, V, M, M, steps)
    return Pm[0].double()


def run3_or_nan(model, pos0, vel0, steps, dev):
    try:
        return run3(model, pos0, vel0, steps, dev)
    except RuntimeError as e:
        log(f"rollout diverged ({str(e)[:80]}), recording non-finite")
        return torch.full((steps + 1, pos0.shape[0], 3), float("nan"), dtype=torch.float64, device=dev)


def cv_err(pos0, vel0, Pt, k, dev):
    return float((Pt[k] - torch.tensor(pos0, device=dev) - torch.tensor(vel0, device=dev) * DT * k).norm(dim=-1).mean())


STAGE_FULL_ITERS = 60000
FAMILIES = {"cold": (8.0, 0.3), "virial": (8.0, 1.0)}


def score(args, dev):
    cand = load_candidate(args, dev)
    ctrl = ss.load(args.control, dev)
    ops.retarget(ctrl, 128, 64.0)
    steps, n = args.score_steps, args.score_n
    per = []
    for seed in HELD_OUT:
        for rep in range(args.score_reps):
            p2, v2 = ss.make_scene(n, 8.0, 0.3, 64.0, seed, rep)
            Pt2, _ = ops.exact_rollout(torch.tensor(p2, device=dev), torch.tensor(v2, device=dev), steps, DT, SUB, EPS)
            Pm2, _, _ = ss.run_model(ctrl, p2, v2, steps, dev, probe=False)
            e2 = (Pm2.double() - Pt2).norm(dim=-1).mean(1)
            row = {"seed": seed, "rep": rep, "d2": {str(k): float(e2[k]) for k in TICKS if k <= steps}, "cv2_20": cv_err(p2, v2, Pt2, 20, dev)}
            for fam, (a, q) in FAMILIES.items():
                p3, v3 = plummer(n, a, q, np.random.default_rng([seed, n, rep]))
                t3, o3 = torch.tensor(p3, device=dev), torch.tensor(v3, device=dev)
                Pt, _ = ops.exact_rollout(t3, o3, steps, DT, SUB, EPS)
                f1, _ = ops.exact_rollout(t3, o3, 20, DT, SUB, EPS)
                f2, _ = ops.exact_rollout(t3, o3, 20, DT, 4 * SUB, EPS)
                Pm = run3_or_nan(cand, p3, v3, steps, dev)
                e3 = (Pm - Pt).norm(dim=-1).mean(1)
                row[fam] = {"d3": {str(k): float(e3[k]) for k in TICKS if k <= steps}, "cv3_20": cv_err(p3, v3, Pt, 20, dev), "floor3_20": float((f1[20] - f2[20]).norm(dim=-1).mean()),
                            "finite": bool(torch.isfinite(Pm).all()), "cell_overflow": cand.cell_overflow}
            per.append(row)
            log(f"seed {seed}/{rep} 2D err@20 {row['d2']['20']:.4g} (cv {row['cv2_20']:.4g}) " + " ".join(f"{f}: 3D err@20 {row[f]['d3']['20']:.4g} (cv {row[f]['cv3_20']:.4g}, floor {row[f]['floor3_20']:.3g}, overflow {row[f]['cell_overflow']})" for f in FAMILIES))
    res = {"per_scene": per, "primary_family": "cold", "plateau": common.plateau(args.out) if os.path.exists(f"{args.out}/val_curve.json") else None}
    if args.mode == "score":
        res["ckpt_file"], res["ckpt_it"] = ckpt_path(args), args.ckpt_it
        res["val_err20"] = val_err(cand, train_scenes(args.val_scenes, 20, VAL_SEED, dev), dev)
        log(f"val err@20 at it {args.ckpt_it}: {res['val_err20']:.5f}")
    ratio2 = float(np.mean([r["d2"]["20"] for r in per]) / np.mean([r["cv2_20"] for r in per]))
    res["ratio2_err20_over_constvel"] = ratio2
    res["ratio2_mean_of_per_scene_ratios"] = float(np.mean([r["d2"]["20"] / r["cv2_20"] for r in per]))
    res["control_md5"] = hashlib.md5(open(args.control, "rb").read()).hexdigest()
    res["n_scenes"] = len(per)
    log(f"2D control ratio to exact (err@20 over zero-force err@20, ratio of means over {len(per)} scenes, N={n}, seeds {list(HELD_OUT)}, control md5 {res['control_md5']}): {ratio2:.6g}; mean of per-scene ratios {res['ratio2_mean_of_per_scene_ratios']:.6g}")
    for fam in FAMILIES:
        r3 = float(np.mean([r[fam]["d3"]["20"] for r in per]) / np.mean([r[fam]["cv3_20"] for r in per]))
        res[fam] = {"ratio3_err20_over_constvel": r3, "ratio3_over_ratio2": r3 / ratio2, "all_finite": all(r[fam]["finite"] for r in per)}
        log(f"3D {fam} ratio to exact (ratio of means) {r3:.6g}; 3D over 2D {r3 / ratio2:.6g}")
    res["ratio3_over_ratio2"] = res["cold"]["ratio3_over_ratio2"]
    json.dump(res, open(f"{args.out}/score.json", "w"), indent=1)
    return res


def timed(fn, steps, reps, dev):
    fn(min(steps, 2))
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(steps)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) / steps * 1000)
    return float(np.median(ts))


def bench(args, dev):
    path = f"{args.out}/bench.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    cand = load_candidate(args, dev)
    ctrl = ss.load(args.control, dev)
    ops.retarget(ctrl, 128, 64.0)
    for n in [int(x) for x in args.bench_ns.split(",")]:
        if str(n) in res:
            continue
        steps, reps = (10, 3) if n <= 10000 else (3, 2)
        rng = np.random.default_rng([4738, n, 0])
        p3, v3 = plummer(n, 8.0, 0.3, rng)
        p2, v2 = ss.make_scene(n, 8.0, 0.3, 64.0, 4738, 0)
        torch.cuda.empty_cache()
        try:
            m3 = timed(lambda k: run3(cand, p3, v3, k, dev), steps, reps, dev)
            m2 = timed(lambda k: ss.run_model(ctrl, p2, v2, k, dev, probe=False), steps, reps, dev)
        except torch.cuda.OutOfMemoryError:
            res["oom_at_n"] = n
            json.dump(res, open(path, "w"), indent=1)
            raise SystemExit(f"CUDA out of memory in bench at n={n}; completed cells kept in {path}")
        res[str(n)] = {"ms_tick_3d": m3, "ms_tick_2d": m2, "ratio": m3 / m2, "ns_body_tick_3d": m3 * 1e6 / n, "ns_body_tick_2d": m2 * 1e6 / n}
        log(f"bench n={n} {res[str(n)]}")
        json.dump(res, open(path, "w"), indent=1)
    ns = sorted(int(k) for k in res if k.isdigit())
    res["cost_ratio_at_1e5"] = res.get("100000", {}).get("ratio")
    res["exponent_3d"] = float(np.polyfit(np.log10(ns), np.log10([res[str(n)]["ms_tick_3d"] for n in ns]), 1)[0]) if len(ns) > 1 else None
    json.dump(res, open(path, "w"), indent=1)
    return res


def verdict(sc, bn, iters):
    rr, c = sc["ratio3_over_ratio2"], bn["cost_ratio_at_1e5"]
    if not (np.isfinite(rr) and sc["cold"]["all_finite"]):
        return "KILL-nonfinite"
    if c is None:
        return "INCONCLUSIVE-no-1e5-timing"
    under = sc["plateau"]["undertrained"]
    if c > 12.0:
        return "KILL"
    if iters < STAGE_FULL_ITERS:
        return "INCONCLUSIVE-undertrained"
    if rr > 4.0:
        return "INCONCLUSIVE-undertrained" if under else "KILL"
    if rr <= 2.0:
        return "PASS"
    return "INCONCLUSIVE-undertrained" if under else "INCONCLUSIVE-between-lines"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "score", "bench", "all"], default="all")
    ap.add_argument("--control", default="kern/E_frozen.pt")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-scenes", type=int, default=48)
    ap.add_argument("--score-n", type=int, default=1000)
    ap.add_argument("--score-reps", type=int, default=4)
    ap.add_argument("--score-steps", type=int, default=100)
    ap.add_argument("--bench-ns", default=",".join(str(n) for n in BENCH_NS))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--heldout", default="9100,9200,9300")
    ap.add_argument("--out", default="results/three_dimensional_sparse_voxel_scatter_field")
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--ckpt-file", default="")
    ap.add_argument("--stop-at", type=int, default=0)
    args = ap.parse_args()
    global HELD_OUT
    HELD_OUT = tuple(int(x) for x in args.heldout.split(","))
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    signal.signal(signal.SIGTERM, on_term)
    if args.mode in ("train", "all") and not os.path.exists(f"{args.out}/train_done.json"):
        train(args, dev)
    if args.mode in ("score", "all"):
        sc = score(args, dev)
    if args.mode in ("bench", "all"):
        bn = bench(args, dev)
    if args.mode == "all":
        v = verdict(sc, bn, json.load(open(f"{args.out}/train_done.json"))["train_iters"])
        json.dump({"verdict": v, "ratio3_over_ratio2": sc["ratio3_over_ratio2"], "cost_ratio_at_1e5": bn["cost_ratio_at_1e5"], "cost_line": "KILL" if bn["cost_ratio_at_1e5"] > 12.0 else "within 12x",
                   "undertrained": sc["plateau"]["undertrained"]}, open(f"{args.out}/summary.json", "w"), indent=1)
        log(f"VERDICT {v}")
        log("DONE")


if __name__ == "__main__":
    main()
