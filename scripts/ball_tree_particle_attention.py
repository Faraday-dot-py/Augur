"""Stage 1 of ball-tree-particle-attention-tier-0: train model/ball_tree_attention.py end to end on the the scatter-field control (ms_kp_pot_v_g128) recipe, score it against the frozen the scatter-field control (ms_kp_pot_v_g128) checkpoint in the same run, and time it from 1e3 to 1e6 bodies.
Modes: train (resumable, SIGTERM-safe, validation curve for the plateau guard), score (err@5/20 at N=1000 cold and warm vs control, in-distribution cfg E, exact-reference floor), bench (ns/body/tick sweep), all.
Usage: PYTHONPATH=. python scripts/ball_tree_particle_attention.py --mode all --control kern/E_frozen.pt --iters 60000 --device cuda --out results/ball_tree_particle_attention
"""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from model import ball_tree_attention as bt
from scripts import scatter_field as sf
from scripts import scatter_scaling as ss
from scripts import scatter_scaling_ops as ops
from scripts import stage1_common as common
from scripts import train_scatter_field as tsf

CFG = tsf.EXPS["E"]
DT, EPS = 0.1, 0.5
HELD_OUT = (9100, 9200, 9300)
VAL_SEED = 9400
INDIST_SEED = 9000
TICKS = (5, 10, 20, 50, 100)
BENCH_NS = (1000, 3162, 10000, 31623, 100000, 316228, 1000000)

STOP = {"flag": False}


def log(s):
    print(s, flush=True)


def on_term(signum, frame):
    STOP["flag"] = True


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def build_candidate(dev):
    return bt.BallTreeAttention(dt=DT).to(dev)


def val_err(model, val, dev):
    with torch.no_grad():
        m, _ = tsf.evaluate_model(model, val, dev, DT, EPS)
    return float(m["err"][19])


def train(args, dev):
    dd = dev if dev.type == "cuda" else None
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = f"{args.ckpt_dir}/ball_tree_particle_attention.pt"
    model = build_candidate(dev)
    data = tsf.make_data(CFG, args.train, args.steps, args.seed, dd, DT, EPS)
    tensors = sf.to_tensors(data, dev)
    val = tsf.make_data(CFG, args.val_scenes, 20, VAL_SEED, dd, DT, EPS)
    n_it, secs = common.train_with_curve(model, tensors, lambda m: val_err(m, val, dev), args, ckpt, args.out, lambda: STOP["flag"], log)
    if STOP["flag"]:
        log("SIGTERM: checkpoint saved, exiting for requeue")
        raise SystemExit(143)
    json.dump({"train_iters": n_it, "train_seconds": secs, "args": vars(args)}, open(f"{args.out}/train_done.json", "w"))
    return model


def load_candidate(args, dev):
    model = build_candidate(dev)
    model.load_state_dict(torch.load(f"{args.ckpt_dir}/ball_tree_particle_attention.pt", map_location=dev, weights_only=False)["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def floor_err(pos0, vel0, dev):
    p64, v64 = torch.tensor(pos0, device=dev), torch.tensor(vel0, device=dev)
    a, _ = ops.exact_rollout(p64, v64, 20, DT, ss.SUB, EPS)
    b, _ = ops.exact_rollout(p64, v64, 20, DT, 4 * ss.SUB, EPS)
    return float((a[20] - b[20]).norm(dim=-1).mean())


def score_family(models, vfac, dev, n, reps, steps):
    per = []
    for seed in HELD_OUT:
        for rep in range(reps):
            pos0, vel0 = ss.make_scene(n, 8.0, vfac, 64.0, seed, rep)
            Pt, _ = ops.exact_rollout(torch.tensor(pos0, device=dev), torch.tensor(vel0, device=dev), steps, DT, ss.SUB, EPS)
            row = {"seed": seed, "rep": rep, "floor_err20": floor_err(pos0, vel0, dev)}
            for name, model in models.items():
                Pm, _, _ = ss.run_model(model, pos0, vel0, steps, dev, probe=False)
                err = (Pm.double() - Pt).norm(dim=-1).mean(1)
                row[name] = {str(k): float(err[k]) for k in TICKS if k <= steps}
                row[name]["finite"] = bool(torch.isfinite(Pm).all())
            per.append(row)
            log(f"vfac {vfac} seed {seed}/{rep} err@20 " + " ".join(f"{n} {row[n]['20']:.4g}" for n in models) + f" floor {row['floor_err20']:.3g}")
    out = {"per_scene": per}
    for name in models:
        out[name] = {str(k): float(np.mean([r[name][str(k)] for r in per])) for k in TICKS if k <= steps}
    out["ratio_err20"] = out["candidate"]["20"] / out["control"]["20"]
    out["ratio_err5"] = out["candidate"]["5"] / out["control"]["5"]
    out["floor_err20"] = float(np.mean([r["floor_err20"] for r in per]))
    out["all_finite"] = all(r["candidate"]["finite"] for r in per)
    return out


def score(args, dev):
    dd = dev if dev.type == "cuda" else None
    cand = load_candidate(args, dev)
    ctrl = ss.load(args.control, dev)
    ops.retarget(ctrl, 128, 64.0)
    models = {"candidate": cand, "control": ctrl}
    res = {"plateau": common.plateau(args.out), "indist": {}}
    data = tsf.make_data(CFG, 48, 100, INDIST_SEED, dd, DT, EPS)
    for name, m in models.items():
        with torch.no_grad():
            r, _ = tsf.evaluate_model(m, data, dev, DT, EPS)
        res["indist"][name] = [r["err"][i] for i in (4, 9, 19)]
        log(f"in-distribution {INDIST_SEED} {name} err@5/10/20 {res['indist'][name]}")
    res["cold"] = score_family(models, 0.3, dev, args.score_n, args.score_reps, args.score_steps)
    res["warm"] = score_family(models, 1.0, dev, args.score_n, args.score_reps, args.score_steps)
    json.dump(res, open(f"{args.out}/score.json", "w"), indent=1)
    return res


def bench(args, dev):
    path = f"{args.out}/bench.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    model = load_candidate(args, dev)
    ns = [int(x) for x in args.bench_ns.split(",")]
    for n in ns:
        if str(n) in res:
            continue
        pos0, vel0 = ss.make_scene(n, 8.0, 0.3, 64.0, 4738, 0)
        steps, reps = (20, 3) if n <= 31623 else ((5, 2) if n <= 316228 else (3, 2))
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            t = ss.timed(model, pos0, vel0, steps, reps, dev) if dev.type == "cuda" else cpu_timed(model, pos0, vel0, steps, reps)
            t["ns_per_body_tick"] = t["tick_ms"] * 1e6 / n
            res[str(n)] = t
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res[str(n)] = {"error": "OOM"}
        log(f"bench n={n} {res[str(n)]}")
        json.dump(res, open(path, "w"), indent=1)
    ok = [n for n in ns if "tick_ms" in res[str(n)]]

    def slope(lo, hi):
        sel = [n for n in ok if lo <= n <= hi]
        if len(sel) < 2:
            return None
        return float(np.polyfit(np.log10(sel), np.log10([res[str(n)]["tick_ms"] for n in sel]), 1)[0])

    res["exponent_1e3_1e6"], res["exponent_1e4_1e6"], res["exponent_1e5_1e6"] = slope(1e3, 1e6), slope(1e4, 1e6), slope(1e5, 1e6)
    res["ns_per_body_tick_1e6"] = res.get("1000000", {}).get("ns_per_body_tick")
    json.dump(res, open(path, "w"), indent=1)
    return res


def cpu_timed(model, pos0, vel0, steps, reps):
    N = pos0.shape[0]
    P, V, M = torch.tensor(pos0, dtype=torch.float32)[None], torch.tensor(vel0, dtype=torch.float32)[None], torch.ones(1, N)
    ts = []
    with torch.no_grad():
        sf.rollout(model, P, V, M, M, 1)
        for _ in range(reps):
            t = time.perf_counter()
            sf.rollout(model, P, V, M, M, steps)
            ts.append((time.perf_counter() - t) / steps * 1000)
    return {"tick_ms": float(np.median(ts)), "peak_mem_mb": None, "timing_steps": steps, "timing_reps": reps}


def verdict(res_score, res_bench):
    r = res_score["cold"]["ratio_err20"]
    under = res_score["plateau"]["undertrained"]
    e, c = res_bench["exponent_1e3_1e6"], res_bench["ns_per_body_tick_1e6"]
    if not (np.isfinite(r) and res_score["cold"]["all_finite"]):
        return "KILL-nonfinite"
    if e is None or c is None:
        return "INCONCLUSIVE-no-1e6-timing"
    cost_kill = e > 1.25
    err_kill = r > 2.0
    if cost_kill or (err_kill and not under):
        return "KILL"
    if err_kill and under:
        return "INCONCLUSIVE-undertrained"
    if r <= 1.15 and e <= 1.10 and c <= 100.0:
        return "PASS"
    if r > 1.15 and under:
        return "INCONCLUSIVE-undertrained"
    return "INCONCLUSIVE-between-lines"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "score", "bench", "all"], default="all")
    ap.add_argument("--control", default="kern/E_frozen.pt")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=30)
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
    ap.add_argument("--out", default="results/ball_tree_particle_attention")
    ap.add_argument("--ckpt-dir", default="ckpt")
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
        res_score = score(args, dev)
    if args.mode in ("bench", "all"):
        res_bench = bench(args, dev)
    if args.mode == "all":
        v = verdict(res_score, res_bench)
        json.dump({"verdict": v, "ratio_err20_cold": res_score["cold"]["ratio_err20"], "exponent_1e3_1e6": res_bench["exponent_1e3_1e6"],
                   "exponent_1e5_1e6": res_bench["exponent_1e5_1e6"], "ns_per_body_tick_1e6": res_bench["ns_per_body_tick_1e6"],
                   "undertrained": res_score["plateau"]["undertrained"]}, open(f"{args.out}/summary.json", "w"), indent=1)
        log(f"VERDICT {v}")
        log("DONE")


if __name__ == "__main__":
    main()
