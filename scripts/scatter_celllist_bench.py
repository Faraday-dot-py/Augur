"""Density-adaptive cell-list pair term (`adapt` opt of ScatterField.pp_acc_adapt) vs the stock dense pair term and the scatter-opt `cell` term.
Phases: parity (forces vs stock/cell), scale (ms/tick vs N, cold collapse), density (pair-term ms vs core density at fixed N).
Usage: PYTHONPATH=. python scripts/scatter_celllist_bench.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --phase parity --out results/scatter_celllist_parity.json
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf
from scripts.scatter_scaling import make_scene

DT = 0.1


def build(ckpt, dev, opts=()):
    m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", DT).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    m.eval()
    for o in opts:
        sf.apply_opt(m, o)
    return m


def sync():
    torch.cuda.synchronize()


def tmed(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def cmp(a, b):
    d = (a - b).abs()
    mx = float(b.abs().max())
    return {"max_abs_diff": float(d.max()), "max_abs": mx, "rel_max": float(d.max()) / max(mx, 1e-30), "n_diff_gt_1e-6rel": int((d.amax(-1) > 1e-6 * mx).sum())}


@torch.no_grad()
def roll(m, pos, vel, steps, times=False):
    n = pos.shape[1]
    mass = torch.ones(1, n, device=pos.device)
    st = (m.init_field(1, pos.device)[0], None)
    p, v = pos, vel
    ts = []
    for _ in range(steps):
        sync()
        t = time.perf_counter()
        p, v, st, _ = m.step(p, v, mass, mass, st)
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    return p, v, ts


def cold(n, extent, dev, sigma=8.0, vfac=0.3):
    p0, v0 = make_scene(n, sigma, vfac, extent, 4738, 0)
    return torch.tensor(p0, dtype=torch.float32, device=dev)[None], torch.tensor(v0, dtype=torch.float32, device=dev)[None]


@torch.no_grad()
def phase_parity(ckpt, dev, out):
    stock, cell, adapt = build(ckpt, dev), build(ckpt, dev, ["cell"]), build(ckpt, dev, ["adapt"])
    res = {}
    cases = []
    for n in (1000, 3000):
        cases.append((f"cold_t0_N{n}", *cold(n, 64.0, dev), None))
        p, v, _ = roll(stock, *cold(n, 64.0, dev), 40)
        cases.append((f"cold_t40_N{n}", p, v, None))
    g = torch.Generator(device=dev).manual_seed(4738)
    dense = torch.randn(1, 3000, 2, device=dev, generator=g) * 0.5
    dense[:, 1500:] = torch.randn(1, 1500, 2, device=dev, generator=g) * 6
    cases.append(("dense_core0.5_halo6_N3000", dense, torch.zeros_like(dense), None))
    tight = torch.randn(1, 2000, 2, device=dev, generator=g) * 0.02
    cases.append(("ultra_core0.02_N2000", tight, torch.zeros_like(tight), None))
    b4 = torch.randn(4, 1200, 2, device=dev, generator=g) * 3
    mk = (torch.rand(4, 1200, device=dev, generator=g) > 0.3).float()
    cases.append(("batch4_masked_N1200", b4, torch.zeros_like(b4), mk))
    far = torch.randn(1, 2000, 2, device=dev, generator=g) * 6
    far[:, :500] += 200.0
    cases.append(("outside_grid_N2000", far, torch.zeros_like(far), None))
    for name, pos, vel, mk in cases:
        B, N = pos.shape[:2]
        mass = torch.ones(B, N, device=dev) * (0.5 + torch.rand(B, N, device=dev, generator=g))
        mk = torch.ones(B, N, device=dev) if mk is None else mk
        a0 = stock.pp_acc(pos, mass, mk)
        r = {"adapt_vs_stock": cmp(adapt.pp_acc(pos, mass, mk), a0), "cell_vs_stock": cmp(cell.pp_acc(pos, mass, mk), a0),
             "adapt_stats": {"levels": adapt.adapt_stats["levels"], "fallback": adapt.adapt_stats["fallback"]}}
        q = (torch.rand(B, N, device=dev, generator=g) > 0.5).float() * mk
        aq = adapt.pp_acc(pos, mass, mk, qmask=q)
        r["adapt_qmask_vs_stock_on_queries"] = cmp(aq * q[..., None], a0 * q[..., None])
        r["adapt_qmask_nonquery_zero"] = bool((aq * (1 - q[..., None])).abs().max() == 0)
        res[name] = r
        print(name, r, flush=True)
    for tag, extent in (("L64", 64.0), ("L256", 256.0)):
        n = 100000
        pos, vel = cold(n, extent, dev)
        p40, v40, _ = roll(adapt, pos, vel, 30)
        for sn, pp in (("t0", pos), ("t30", p40)):
            mass, mk = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
            a = adapt.pp_acc(pp, mass, mk)
            c = cell.pp_acc(pp, mass, mk)
            res[f"N1e5_{tag}_{sn}"] = {"adapt_vs_cell": cmp(a, c), "adapt_stats": {"levels": adapt.adapt_stats["levels"], "fallback": adapt.adapt_stats["fallback"]}}
            print(f"N1e5 {tag} {sn}", res[f"N1e5_{tag}_{sn}"], flush=True)
    json.dump(res, open(out, "w"))


@torch.no_grad()
def phase_scale(ckpt, dev, out):
    res = {}
    for tag, extent, grid in (("L64", 64.0, 128), ("L256", 256.0, 512)):
        for n in (1000, 3162, 10000, 31623, 100000):
            for opt in ("cell", "adapt"):
                m = build(ckpt, dev, [opt])
                ops.retarget(m, grid, extent)
                pos, vel = cold(n, extent, dev)
                steps = 40 if n >= 10000 else 40
                roll(m, pos, vel, 2)
                p, v, ts = roll(m, pos, vel, steps)
                mass, mk = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
                r = {"n": n, "extent": extent, "tick_ms": ts, "mean_ms_1_10": float(np.mean(ts[:10])), "mean_ms_31_40": float(np.mean(ts[30:40])),
                     "pp_ms_t0": tmed(lambda: m.pp_acc(pos, mass, mk)), "pp_ms_t40": tmed(lambda: m.pp_acc(p, mass, mk)),
                     "frac_outside_t40": float(((p.abs() > extent / 2).any(-1)).float().mean())}
                if opt == "adapt":
                    m.pp_acc(p, mass, mk)
                    r["adapt_stats_t40"] = {"levels": m.adapt_stats["levels"], "fallback": m.adapt_stats["fallback"]}
                res[f"{tag}_{opt}_{n}"] = r
                print(tag, opt, n, {k: v for k, v in r.items() if k != "tick_ms"}, flush=True)
                json.dump(res, open(out, "w"))
                del m
                torch.cuda.empty_cache()
        for n in (316228, 1000000):
            m = build(ckpt, dev, ["adapt"])
            ops.retarget(m, grid, extent)
            pos, vel = cold(n, extent, dev)
            p, v, ts = roll(m, pos, vel, 20)
            mass, mk = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
            r = {"n": n, "extent": extent, "tick_ms": ts, "mean_ms_1_10": float(np.mean(ts[:10])), "mean_ms_11_20": float(np.mean(ts[10:20])),
                 "pp_ms_t0": tmed(lambda: m.pp_acc(pos, mass, mk), reps=1, warm=0), "pp_ms_t20": tmed(lambda: m.pp_acc(p, mass, mk), reps=1, warm=0)}
            m.pp_acc(p, mass, mk)
            r["adapt_stats_t20"] = {"levels": m.adapt_stats["levels"], "fallback": m.adapt_stats["fallback"]}
            res[f"{tag}_adapt_{n}"] = r
            print(tag, "adapt", n, {k: v for k, v in r.items() if k != "tick_ms"}, flush=True)
            json.dump(res, open(out, "w"))
            del m
            torch.cuda.empty_cache()


@torch.no_grad()
def phase_density(ckpt, dev, out):
    cell, adapt = build(ckpt, dev, ["cell"]), build(ckpt, dev, ["adapt"])
    res = {}
    n = 100000
    for sigma in (32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.03):
        g = torch.Generator(device=dev).manual_seed(4738)
        pos = torch.randn(1, n, 2, device=dev, generator=g) * sigma
        mass, mk = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
        c = ((pos.reshape(-1, 2) + 32.0) / 2.0).floor().clamp(0, 31).long()
        occ = torch.bincount(c[:, 1] * 32 + c[:, 0], minlength=1024)
        M = int(occ.max())
        r = {"sigma": sigma, "peak_density_per_unit2": float(n / (2 * np.pi * sigma ** 2)), "max_cell_occupancy": M,
             "adapt_ms": tmed(lambda: adapt.pp_acc(pos, mass, mk))}
        adapt.pp_acc(pos, mass, mk)
        r["adapt_stats"] = {"levels": adapt.adapt_stats["levels"], "fallback": adapt.adapt_stats["fallback"]}
        if n * 9 * M < 3e11:
            r["cell_ms"] = tmed(lambda: cell.pp_acc(pos, mass, mk), reps=1, warm=0)
            r["adapt_vs_cell"] = cmp(adapt.pp_acc(pos, mass, mk), cell.pp_acc(pos, mass, mk))
        else:
            r["cell_ms"] = None
            r["cell_skipped_candidates"] = n * 9 * M
        if sigma <= 0.25:
            adapt.adapt_lmax = 14
            r["adapt_lmax14_ms"] = tmed(lambda: adapt.pp_acc(pos, mass, mk))
            r["adapt_lmax14_vs_cell"] = cmp(adapt.pp_acc(pos, mass, mk), cell.pp_acc(pos, mass, mk))
            r["adapt_lmax14_stats"] = {"levels": adapt.adapt_stats["levels"], "fallback": adapt.adapt_stats["fallback"]}
            adapt.adapt_lmax = 8
        res[f"sigma{sigma:g}"] = r
        print(r, flush=True)
        json.dump(res, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phase", required=True, choices=["parity", "scale", "density"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    print(torch.cuda.get_device_name(dev), flush=True)
    {"parity": phase_parity, "scale": phase_scale, "density": phase_density}[a.phase](a.ckpt, dev, a.out)


if __name__ == "__main__":
    main()
