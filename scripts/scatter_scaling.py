"""Zero-shot scaling tests of the trained scatter-field model (no retraining): particle count N, world extent, grid cells, cell size.

Scene: N unit-mass bodies, Gaussian cluster of std sigma clipped to +-0.45*extent, per-component speed vfac*sqrt(N/(4 sigma)), zero net momentum (the scatter_bh scene, non-relativistic).
Reference: exact softened all-pairs fp64 sim (4 substeps/tick, eps 0.5, chunked on GPU). Model: Verlet scatter-field, dt 0.1, same weights; only (grid, extent) are changed.
Phases: nsweep (N sweep at fixed world), worlds (R resolution, D domain, S scaled-scene, S2, J joint const-density), stock (stock vs chunked pair term: time/memory/OOM), comp (per-component time).
Usage: PYTHONPATH=. python scripts/scatter_scaling.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --phase nsweep --out results/scatter_scaling_nsweep.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

TICKS = (5, 10, 20, 50, 100)
PROBE = (1, 10, 50)
SEEDS = (4738, 9100, 9200, 9300)
NS = [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000, 1778, 3162, 5623, 10000, 17783, 31623, 56234, 100000]
EPS, DT, SUB = 0.5, 0.1, 4
PP = None


def sync():
    torch.cuda.synchronize()


def load(ckpt, dev, chunked=True):
    m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", DT).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    m.eval()
    mode = {True: "chunked", False: "stock"}.get(chunked, chunked) if PP is None or chunked is False else PP
    if mode == "cell":
        sf.apply_opt(m, "cell")
        return m
    return ops.chunked_pp(m) if mode == "chunked" else m


def scenes_for(n):
    reps = 4 if n <= 1000 else (2 if n <= 10000 else 1)
    seeds = SEEDS if n < 31623 else SEEDS[:1]
    return [(s, r) for s in seeds for r in range(reps)] if n <= 10000 else [(s, 0) for s in seeds]


def make_scene(n, sigma, vfac, extent, seed, rep):
    rng = np.random.default_rng([seed, n, rep])
    lim = 0.45 * extent
    pos = np.clip(rng.normal(0.0, sigma, (n, 2)), -lim, lim)
    vel = rng.normal(0.0, vfac * np.sqrt(n / (4 * sigma)), (n, 2))
    vel -= vel.mean(0)
    return pos, vel


def stats(P, V):
    c = P.mean(1, keepdim=True)
    r = P - c
    L = (r[..., 0] * V[..., 1] - r[..., 1] * V[..., 0]).sum(1)
    mom = V.sum(1).norm(dim=-1)
    return L, mom


def run_model(model, pos0, vel0, steps, dev, probe=True):
    B, N = 1, pos0.shape[0]
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, N, device=dev)
    mask = torch.ones(1, N, device=dev)
    field = model.init_field(1, dev)
    ps, vs, probes = [pos[0]], [vel[0]], {}
    with torch.no_grad():
        for t in range(1, steps + 1):
            pos, vel, field, _ = model.step(pos, vel, mass, mask, field)
            ps.append(pos[0])
            vs.append(vel[0])
            if probe and t in PROBE:
                a_m = field[1][0].double()
                p64 = pos[0].double()
                a_t = ops.exact_acc(p64, EPS)
                rr = (p64 - p64.mean(0)).norm(dim=-1)
                order = rr.argsort()
                an = a_t.norm(dim=-1)
                rel = (a_m - a_t).norm(dim=-1) / an.clamp_min(1e-12)
                pd = {"median_rel": float(rel.median()), "wmean_rel": float((a_m - a_t).norm(dim=-1).sum() / an.sum()),
                      "cos": float(((a_m * a_t).sum(-1) / (a_m.norm(dim=-1) * an).clamp_min(1e-12)).mean())}
                for name, sl in (("core", slice(0, N // 3)), ("mid", slice(N // 3, 2 * N // 3)), ("halo", slice(2 * N // 3, N))):
                    ii = order[sl]
                    pd[f"wmean_rel_{name}"] = float((a_m[ii] - a_t[ii]).norm(dim=-1).sum() / an[ii].sum().clamp_min(1e-12))
                    pd[f"amag_{name}"] = float(an[ii].mean())
                old = model.pp_acc
                ops.no_pp(model)
                a_np, _ = model.force(pos, vel, mass, mask, field[0])
                model.pp_acc = old
                a_np = a_np[0].double()
                pd["wmean_rel_nopp"] = float((a_np - a_t).norm(dim=-1).sum() / an.sum())
                probes[str(t)] = pd
    return torch.stack(ps), torch.stack(vs), probes


def timed(model, pos0, vel0, steps, reps, dev):
    N = pos0.shape[0]
    P = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    V = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    M = torch.ones(1, N, device=dev)
    sf.rollout(model, P, V, M, M, min(steps, 3))
    sync()
    ts, peaks = [], []
    for _ in range(reps):
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        sync()
        t = time.perf_counter()
        sf.rollout(model, P, V, M, M, steps)
        sync()
        ts.append((time.perf_counter() - t) / steps * 1000)
        peaks.append((torch.cuda.max_memory_allocated(dev) - base) / 2 ** 20)
    return {"tick_ms": float(np.median(ts)), "peak_mem_mb": float(max(peaks)), "timing_steps": steps, "timing_reps": reps}


def point(model, n, sigma, vfac, grid, extent, dev, steps=100, time_it=True, label=None):
    ops.retarget(model, grid, extent)
    errs, cverrs, per = [], [], []
    res = {"n": n, "sigma": sigma, "vfac": vfac, "grid": grid, "extent": extent, "h": extent / grid, "label": label}
    first = None
    t_truth = None
    for i, (seed, rep) in enumerate(scenes_for(n)):
        pos0, vel0 = make_scene(n, sigma, vfac, extent, seed, rep)
        p64 = torch.tensor(pos0, device=dev)
        v64 = torch.tensor(vel0, device=dev)
        sync()
        t0 = time.perf_counter()
        Pt, Vt = ops.exact_rollout(p64, v64, steps, DT, SUB, EPS)
        sync()
        t_truth = time.perf_counter() - t0
        try:
            Pm, Vm, probes = run_model(model, pos0, vel0, steps, dev)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            res["error"] = "OOM in model rollout"
            return res
        Pm2, _, _ = run_model(model, pos0, vel0, steps, dev, probe=False)
        selferr = (Pm2.double() - Pm.double()).norm(dim=-1).mean(1)
        del Pm2
        Pm, Vm = Pm.double(), Vm.double()
        fin = torch.isfinite(Pm).all(-1).all(-1)
        bad = (~fin).nonzero()
        err = (Pm - Pt).norm(dim=-1).mean(1)
        cv = (Pt[0][None] + Vt[0][None] * DT * torch.arange(steps + 1, device=dev, dtype=torch.float64)[:, None, None] - Pt).norm(dim=-1).mean(1)
        Lm, pm = stats(Pm, Vm)
        Lt, pt = stats(Pt, Vt)
        e0 = ops.exact_energy(Pt[0], Vt[0], EPS)
        em = ops.exact_energy(Pm[-1], Vm[-1], EPS) if bool(fin[-1]) else float("nan")
        et = ops.exact_energy(Pt[-1], Vt[-1], EPS)
        lim = extent / 2
        d = {"seed": seed, "rep": rep, "err": {str(k): float(err[k]) for k in TICKS}, "constvel": {str(k): float(cv[k]) for k in TICKS},
             "first_nonfinite": int(bad[0]) if len(bad) else None, "selfnoise": {str(k): float(selferr[k]) for k in TICKS},
             "dE_rel_model": abs(em - e0) / max(abs(e0), 1e-9), "dE_rel_truth": abs(et - e0) / max(abs(e0), 1e-9),
             "dP_model": float(pm[-1] - pm[0]), "dP_truth": float(pt[-1] - pt[0]),
             "dL_rel_model": float(abs(Lm[-1] - Lm[0]) / Lm[0].abs().clamp_min(1e-3)), "dL_rel_truth": float(abs(Lt[-1] - Lt[0]) / Lt[0].abs().clamp_min(1e-3)),
             "rms_r0": float((Pt[0] - Pt[0].mean(0)).norm(dim=-1).pow(2).mean().sqrt()),
             "rms_r_model": float((Pm[-1] - Pm[-1].mean(0)).norm(dim=-1).pow(2).mean().sqrt()),
             "rms_r_truth": float((Pt[-1] - Pt[-1].mean(0)).norm(dim=-1).pow(2).mean().sqrt()),
             "frac_out_model": float((Pm[-1].abs() > lim).any(-1).double().mean()), "frac_out_truth": float((Pt[-1].abs() > lim).any(-1).double().mean()),
             "truth_s": t_truth, "probes": probes}
        per.append(d)
        if first is None:
            first = (pos0, vel0)
        del Pt, Vt, Pm, Vm
        print(f"  n={n} grid={grid} L={extent} seed={seed}/{rep} err@5/20/100 {d['err']['5']:.4g} {d['err']['20']:.4g} {d['err']['100']:.4g} "
              f"cv@20 {d['constvel']['20']:.4g} nonfinite {d['first_nonfinite']} truth {t_truth:.1f}s", flush=True)
    res["per_scene"] = per
    res["err"] = {str(k): float(np.mean([d["err"][str(k)] for d in per])) for k in TICKS}
    res["selfnoise"] = {str(k): float(np.mean([d["selfnoise"][str(k)] for d in per])) for k in TICKS}
    res["constvel"] = {str(k): float(np.mean([d["constvel"][str(k)] for d in per])) for k in TICKS}
    for k in ("dE_rel_model", "dE_rel_truth", "dP_model", "dP_truth", "dL_rel_model", "dL_rel_truth", "frac_out_model", "frac_out_truth", "rms_r_model", "rms_r_truth"):
        res[k] = float(np.mean([d[k] for d in per]))
    nf = [d["first_nonfinite"] for d in per if d["first_nonfinite"] is not None]
    res["first_nonfinite"] = min(nf) if nf else None
    if time_it:
        steps_t, reps = (100, 3) if n < 10000 else ((20, 2) if n < 31623 else (10, 1))
        try:
            res["timing"] = timed(model, first[0], first[1], steps_t, reps, dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res["timing"] = {"error": "OOM"}
    return res


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save_json(path, d):
    json.dump(d, open(path, "w"), indent=1)


def phase_nsweep(model, dev, out, ns):
    res = load_json(out)
    for fam, vfac in (("cold", 0.3), ("warm", 1.0)):
        for n in ns:
            key = f"{fam}_{n}"
            if key in res:
                continue
            print(f"== nsweep {key}", flush=True)
            res[key] = point(model, n, 8.0, vfac, 128, 64.0, dev, label=key)
            save_json(out, res)
            torch.cuda.empty_cache()


def world_points():
    pts = []
    for g in (32, 64, 128, 256, 512):
        pts.append(("R", f"R_G{g}", 300, 8.0, g, 64.0))
    for L in (32, 48, 64, 96, 128, 256, 512):
        pts.append(("D", f"D_L{L}", 300, 8.0, int(2 * L), float(L)))
    for L in (16, 32, 64, 128, 256, 512):
        pts.append(("S", f"S_L{L}", 300, L / 8, 128, float(L)))
    for L in (32, 64, 128, 256, 512):
        pts.append(("S2", f"S2_L{L}", 300, L / 8, int(2 * L), float(L)))
    for L in (16, 32, 64, 128, 256, 512, 1024):
        pts.append(("J", f"J_L{L}", int(round(300 * (L / 64) ** 2)), L / 8, int(2 * L), float(L)))
    return pts


def phase_worlds(model, dev, out):
    res = load_json(out)
    for fam, key, n, sigma, g, L in world_points():
        if key in res:
            continue
        print(f"== worlds {key} N={n} sigma={sigma} G={g} L={L}", flush=True)
        res[key] = point(model, n, sigma, 0.3, g, L, dev, label=key)
        res[key]["family"] = fam
        save_json(out, res)
        torch.cuda.empty_cache()


def phase_fprobe(ckpt, dev, out):
    model = load(ckpt, dev, True)
    res = {}
    for ncount in (75, 300, 1200, 4800, 19200):
        for sigma in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
            L = 8 * sigma
            if sigma == 128.0 and ncount > 4800:
                continue
            ops.retarget(model, int(2 * L), L)
            pos0, vel0 = make_scene(ncount, sigma, 0.3, L, 4738, 0)
            _, _, pr = run_model(model, pos0, vel0, 1, dev)
            res[f"N{ncount}_s{sigma:g}"] = {"n": ncount, "sigma": sigma, "sigma_cells": sigma / 0.5, "extent": L, "h": 0.5, "probe": pr["1"]}
            print(f"N {ncount} sigma {sigma} ({2 * sigma} cells) tick1 wmean_rel {pr['1']['wmean_rel']:.3f} core/mid/halo {pr['1']['wmean_rel_core']:.3f}/{pr['1']['wmean_rel_mid']:.3f}/{pr['1']['wmean_rel_halo']:.3f} cos {pr['1']['cos']:.3f}", flush=True)
            torch.cuda.empty_cache()
    for h in (0.125, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0):
        L = 64.0
        g = int(round(L / h / 32)) * 32
        ops.retarget(model, g, L)
        pos0, vel0 = make_scene(300, 8.0, 0.3, L, 4738, 0)
        _, _, pr = run_model(model, pos0, vel0, 1, dev)
        res[f"h{h:g}"] = {"h": L / g, "grid": g, "probe": pr["1"]}
        print(f"h {L / g:.3f} G {g} tick1 wmean_rel {pr['1']['wmean_rel']:.3f} nopp {pr['1']['wmean_rel_nopp']:.3f} cos {pr['1']['cos']:.3f}", flush=True)
    save_json(out, res)


def phase_stock(ckpt, dev, out):
    res = load_json(out)
    for variant in ("stock", "chunked"):
        model = load(ckpt, dev, chunked=(variant == "chunked"))
        ops.retarget(model, 128, 64.0)
        for n in NS:
            if n < 100:
                continue
            key = f"{variant}_{n}"
            if key in res:
                continue
            pos0, vel0 = make_scene(n, 8.0, 0.3, 64.0, 4738, 0)
            steps, reps = (20, 3) if n < 10000 else (5, 1)
            try:
                res[key] = timed(model, pos0, vel0, steps, reps, dev)
            except torch.cuda.OutOfMemoryError:
                res[key] = {"error": "OOM"}
                torch.cuda.empty_cache()
            res[key]["n"] = n
            print(key, res[key], flush=True)
            save_json(out, res)
            torch.cuda.empty_cache()


@torch.no_grad()
def phase_comp(ckpt, dev, out):
    model = load(ckpt, dev, chunked=True)
    ops.retarget(model, 128, 64.0)
    res = {}
    for n in (1000, 10000, 100000):
        pos0, vel0 = make_scene(n, 8.0, 0.3, 64.0, 4738, 0)
        pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
        vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
        mass = torch.ones(1, n, device=dev)
        field = model.init_field(1, dev)[0]
        times = {k: [] for k in ("scatter", "kernel_fft", "unet", "gather", "pp", "momfix_other", "total")}
        for rep in range(4):
            sync()
            t0 = time.perf_counter()
            x = model.scatter(pos, vel, mass, mass)
            sync()
            t1 = time.perf_counter()
            ak = model.kernel_acc(x[:, :1])
            sync()
            t2 = time.perf_counter()
            x = torch.cat([x, ak, field], 1)
            f = model.net(x)
            acc = model.neg_grad(f[:, :1]) + ak
            sync()
            t3 = time.perf_counter()
            a = model.gather(acc, pos)
            sync()
            t4 = time.perf_counter()
            a = a + model.pp_acc(pos, mass, mass)
            sync()
            t5 = time.perf_counter()
            if rep:
                for k, v in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, 0.0, t5 - t0)):
                    times[k].append(v * 1000)
        res[str(n)] = {k: float(np.median(v)) for k, v in times.items()}
        print(n, res[str(n)], flush=True)
        torch.cuda.empty_cache()
    save_json(out, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phase", required=True, choices=["nsweep", "worlds", "stock", "comp", "equiv", "dump", "fprobe", "equivcell"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=100000)
    ap.add_argument("--pp", default=None, choices=[None, "chunked", "cell"])
    args = ap.parse_args()
    global PP
    PP = args.pp
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(dev), flush=True)
    if args.phase == "stock":
        phase_stock(args.ckpt, dev, args.out)
    elif args.phase == "comp":
        phase_comp(args.ckpt, dev, args.out)
    elif args.phase == "fprobe":
        phase_fprobe(args.ckpt, dev, args.out)
    elif args.phase == "equivcell":
        PP_SAVE = PP
        a = load(args.ckpt, dev, "chunked")
        b = load(args.ckpt, dev, "cell")
        out = {}
        for n in (600, 2000, 10000, 100000):
            pos0, vel0 = make_scene(n, 8.0, 0.3, 64.0, 4738, 0)
            P0 = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
            ones = torch.ones(1, n, device=dev)
            with torch.no_grad():
                fa, fb = a.pp_acc(P0, ones, ones), b.pp_acc(P0, ones, ones)
            out[str(n)] = {"max_abs_diff": float((fa - fb).abs().max()), "max_abs": float(fa.abs().max()), "rel_l2": float((fa - fb).norm() / fa.norm())}
            print("chunked vs cell pp_acc", n, out[str(n)], flush=True)
        save_json(args.out, out)
    elif args.phase == "dump":
        model = load(args.ckpt, dev, True)
        for n in (1000, 10000, 100000):
            ops.retarget(model, 128, 64.0)
            pos0, vel0 = make_scene(n, 8.0, 0.3, 64.0, 4738, 0)
            Pt, _ = ops.exact_rollout(torch.tensor(pos0, device=dev), torch.tensor(vel0, device=dev), 100, DT, SUB, EPS)
            Pm, _, _ = run_model(model, pos0, vel0, 100, dev, probe=False)
            ticks = [0, 5, 10, 20, 50, 100]
            np.savez(f"{args.out}_{n}.npz", ticks=np.array(ticks), model=Pm[ticks].cpu().numpy(), truth=Pt[ticks].float().cpu().numpy(),
                     err=(Pm.double() - Pt).norm(dim=-1).mean(1).cpu().numpy(), extent=64.0)
            print("dumped", n, flush=True)
            del Pt, Pm
            torch.cuda.empty_cache()
    elif args.phase == "equiv":
        a, b = load(args.ckpt, dev, False), load(args.ckpt, dev, True)
        out = {}
        for n in (2, 17, 300, 2000):
            pos0, vel0 = make_scene(n, 8.0, 0.3, 64.0, 4738, 0)
            pa, _, _ = run_model(a, pos0, vel0, 20, dev, probe=False)
            pb, _, _ = run_model(b, pos0, vel0, 20, dev, probe=False)
            dp = (pa - pb).abs().amax(dim=(1, 2))
            P0 = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
            ones = torch.ones(1, n, device=dev)
            with torch.no_grad():
                fa = a.pp_acc(P0, ones, ones)
                fb = b.pp_acc(P0, ones, ones)
            out[str(n)] = {"max_dpos_ticks_1_5_20": [float(dp[t]) for t in (1, 5, 20)], "pp_acc_max_abs_diff": float((fa - fb).abs().max()), "pp_acc_max_abs": float(fa.abs().max())}
            print("stock vs chunked", n, out[str(n)], flush=True)
        save_json(args.out, out)
    else:
        model = load(args.ckpt, dev, True)
        if args.phase == "nsweep":
            phase_nsweep(model, dev, args.out, [n for n in NS if n <= args.max_n])
        else:
            phase_worlds(model, dev, args.out)


if __name__ == "__main__":
    main()
