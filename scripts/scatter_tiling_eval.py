"""Evaluation of tiled ScatterField copies against one big single grid (see docs/debugging/scatter-tiling.md).

Phases: force (one step from shared single-grid states: acceleration error by distance to the nearest tile seam),
roll (20-tick rollouts: err@k vs exact truth and vs the single grid, momentum drift, by seam class),
scale (cost and communication for 1, 4, 16 tiles at constant density).
Usage: PYTHONPATH=. python scripts/scatter_tiling_eval.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --phase force --out results/scatter_tiling_force.json
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
from scripts.scatter_tiling import TiledScatter

EPS, DT, SUB = 0.5, 0.1, 4
DEV_SEEDS, HELD_SEEDS = (4738,), (9000, 12000)
REPS = 3
FAMILIES = {"cold300": (300, 8.0, 0.3), "warm300": (300, 8.0, 1.0), "warm1200": (1200, 16.0, 1.0)}
# name: (nt, tile, halo, ghosts, field_x, far)
VARIANTS = {
    "naive": (2, 64, 0, False, False, False),
    "g16": (2, 64, 16, True, False, False),
    "g16_fx": (2, 64, 16, True, True, False),
    "g16_fx_far": (2, 64, 16, True, True, True),
    "g8_fx": (2, 64, 8, True, True, False),
    "g8_fx_far": (2, 64, 8, True, True, True),
    "g32_fx": (2, 64, 32, True, True, False),
    "t32_naive": (4, 32, 0, False, False, False),
    "t32_g16_fx": (4, 32, 16, True, True, False),
    "t32_g16_fx_far": (4, 32, 16, True, True, True),
}
CLASSES = ("seam<2", "seam2-4", "seam4-8", "seam<8", "int8-16", "int>16", "interior", "outer<8")


def load(ckpt, dev):
    m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", DT).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    return m.eval()


def make_tiled(model, spec):
    nt, tile, halo, g, fx, far = spec
    return TiledScatter(model, nt, tile, halo, ghosts=g, field_x=fx, far=far)


def class_masks(ds, dout):
    ok = dout >= 8
    return {"seam<2": (ds < 2) & ok, "seam2-4": (ds >= 2) & (ds < 4) & ok, "seam4-8": (ds >= 4) & (ds < 8) & ok,
            "seam<8": (ds < 8) & ok, "int8-16": (ds >= 8) & (ds < 16) & ok, "int>16": (ds >= 16) & ok,
            "interior": (ds >= 8) & ok, "outer<8": ~ok}


def rel_by_class(a, ref, masks):
    e, n = (a - ref).norm(dim=-1), ref.norm(dim=-1)
    out = {}
    for k, mk in masks.items():
        out[k] = {"n": int(mk.sum()), "rel": float(e[mk].sum() / n[mk].sum().clamp_min(1e-12)) if mk.any() else None}
    return out


def single_setup(model, extent=128.0):
    ops.retarget(model, round(extent / 0.5), extent)


def scenes(fam, seeds):
    n, sigma, vfac = FAMILIES[fam]
    return [(s, r, make_scene(n, sigma, vfac, 128.0, s, r)) for s in seeds for r in range(REPS)]


@torch.no_grad()
def phase_force(model, dev, out):
    res = {}
    ticks = (0, 5, 10, 20)
    for fam in FAMILIES:
        for seeds_name, seeds in (("dev", DEV_SEEDS), ("held", HELD_SEEDS)):
            rows = []
            for seed, rep, (p0, v0) in scenes(fam, seeds):
                n = p0.shape[0]
                single_setup(model)
                pos = torch.tensor(p0, dtype=torch.float32, device=dev)
                vel = torch.tensor(v0, dtype=torch.float32, device=dev)
                mass = torch.ones(n, device=dev)
                msk = torch.ones(1, n, device=dev)
                st = (model.init_field(1, dev)[0], None)
                states = {}
                p, v = pos[None], vel[None]
                for t in range(max(ticks) + 1):
                    if t in ticks:
                        states[t] = (p[0].clone(), v[0].clone(), st[0].clone(), None if st[1] is None else st[1][0].clone())
                    p, v, st, _ = model.step(p, v, mass[None], msk, st)
                for t in ticks:
                    pos_t, vel_t, f_t, ap_t = states[t]
                    single_setup(model)
                    ps, vs, (fs, a_s), _ = model.step(pos_t[None], vel_t[None], mass[None], msk, (f_t, None if ap_t is None else ap_t[None]))
                    a_s = a_s[0]
                    pn = ps[0]
                    a_x = ops.exact_acc(pn.double(), EPS).float()
                    row = {"seed": seed, "rep": rep, "tick": t}
                    for name, spec in VARIANTS.items():
                        T = make_tiled(model, spec)
                        ft = T.slice_field(f_t[0])
                        ap = ap_t
                        pt, vt, (_, a_t) = T.step(pos_t, vel_t, mass, (ft, ap))
                        ds, dout = T.seam_dist(pn)
                        mk = class_masks(ds, dout)
                        row[name] = {"vs_single": rel_by_class(a_t, a_s, mk), "vs_exact": rel_by_class(a_t, a_x, mk),
                                     "pos_diff_max": float((pt - ps[0]).norm(dim=-1).max()), "ghosts": T.stats.get("ghosts_mean")}
                        if name == "g16_fx":
                            row["single_vs_exact_by_g16_class"] = rel_by_class(a_s, a_x, mk)
                    rows.append(row)
                single_setup(model)
                print(f"force {fam} {seeds_name} seed {seed}/{rep} done", flush=True)
            res[f"{fam}/{seeds_name}"] = rows
    json.dump(res, open(out, "w"))


def pos_err(P, Pt, k):
    return float((P[k].double() - Pt[k]).norm(dim=-1).mean())


@torch.no_grad()
def phase_roll(model, dev, out, variants, steps=20):
    res = {}
    for fam in FAMILIES:
        for seeds_name, seeds in (("dev", DEV_SEEDS), ("held", HELD_SEEDS)):
            rows = []
            for seed, rep, (p0, v0) in scenes(fam, seeds):
                n = p0.shape[0]
                Pt, Vt = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), steps, DT, SUB, EPS)
                pos = torch.tensor(p0, dtype=torch.float32, device=dev)
                vel = torch.tensor(v0, dtype=torch.float32, device=dev)
                mass = torch.ones(n, device=dev)
                single_setup(model)
                msk = torch.ones(1, n, device=dev)
                st = (model.init_field(1, dev)[0], None)
                p, v = pos[None], vel[None]
                Ps, Vs = [p[0]], [v[0]]
                for _ in range(steps):
                    p, v, st, _ = model.step(p, v, mass[None], msk, st)
                    Ps.append(p[0])
                    Vs.append(v[0])
                runs = {"single": (Ps, Vs)}
                for name in variants:
                    T = make_tiled(model, VARIANTS[name])
                    ts = (T.init_field(dev), None)
                    p, v = pos, vel
                    Pl, Vl = [p], [v]
                    for _ in range(steps):
                        p, v, ts = T.step(p, v, mass, ts)
                        Pl.append(p)
                        Vl.append(v)
                    runs[name] = (Pl, Vl)
                single_setup(model)
                row = {"seed": seed, "rep": rep, "constvel20": float((Pt[0] + Vt[0] * DT * steps - Pt[steps]).norm(dim=-1).mean())}
                Psd = torch.stack(runs["single"][0]).double()
                for name, (Pl, Vl) in runs.items():
                    P = torch.stack(Pl).double()
                    V = torch.stack(Vl).double()
                    d = {"err": {str(k): pos_err(P, Pt, k) for k in (5, 10, 20)},
                         "diff_single": {str(k): float((P[k] - Psd[k]).norm(dim=-1).mean()) for k in (5, 10, 20)},
                         "dP": float((V[steps].sum(0).norm() - V[0].sum(0).norm()).abs()), "P20": float(V[steps].sum(0).norm()),
                         "finite": bool(torch.isfinite(P).all())}
                    spec = VARIANTS.get(name, VARIANTS["g16_fx"])
                    ds, dout = TiledScatter(model, spec[0], spec[1], spec[2], ghosts=False, field_x=False).seam_dist(torch.tensor(p0, device=dev))
                    mk = class_masks(ds, dout)
                    e20 = (P[steps] - Pt[steps]).norm(dim=-1)
                    d["err20_by_class"] = {k: float(e20[m].mean()) if m.any() else None for k, m in mk.items() if k in ("seam<8", "interior")}
                    row[name] = d
                rows.append(row)
                print(f"roll {fam} {seeds_name} seed {seed}/{rep}: " + " ".join(f"{k}={v['err']['20']:.3g}" for k, v in row.items() if isinstance(v, dict) and "err" in v), flush=True)
            res[f"{fam}/{seeds_name}"] = rows
    json.dump(res, open(out, "w"))


@torch.no_grad()
def phase_big(model, dev, out, steps=20):
    """16 tiles of 64 (extent 256) vs one single grid (G=512), constant-density J scene; no tiling-free truth claim: the single grid itself has a wrong far field here."""
    L, n, sigma = 256.0, 4800, 32.0
    names = {"g16_fx": (4, 64, 16, True, True, False), "g16_fx_far": (4, 64, 16, True, True, True)}
    rows = []
    for seed in (4738, 9000, 12000):
        p0, v0 = make_scene(n, sigma, 1.0, L, seed, 0)
        Pt, Vt = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), steps, DT, SUB, EPS)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)
        mass = torch.ones(n, device=dev)
        msk = torch.ones(1, n, device=dev)
        ops.retarget(model, 512, L)
        st = (model.init_field(1, dev)[0], None)
        p, v = pos[None], vel[None]
        Ps, states = [p[0]], {}
        for t in range(steps):
            if t in (0, 10):
                states[t] = (p[0].clone(), v[0].clone(), st[0].clone(), None if st[1] is None else st[1][0].clone())
            p, v, st, _ = model.step(p, v, mass[None], msk, st)
            Ps.append(p[0])
        Ps = torch.stack(Ps).double()
        row = {"seed": seed, "single_err": {str(k): pos_err(Ps, Pt, k) for k in (5, 10, 20)}, "dP_single": float((v[0].sum(0).norm() - vel.sum(0).norm()).abs())}
        for name, spec in names.items():
            T = make_tiled(model, spec)
            r = {}
            for t, (pos_t, vel_t, f_t, ap_t) in states.items():
                ops.retarget(model, 512, L)
                ps, _, (_, a_s), _ = model.step(pos_t[None], vel_t[None], mass[None], msk, (f_t, None if ap_t is None else ap_t[None]))
                pt, _, (_, a_t) = T.step(pos_t, vel_t, mass, (T.slice_field(f_t[0]), ap_t))
                ds, dout = T.seam_dist(ps[0])
                r[f"force_tick{t}"] = rel_by_class(a_t, a_s[0], class_masks(ds, dout))
            ts = (T.init_field(dev), None)
            p, v = pos, vel
            Pl = [p]
            for _ in range(steps):
                p, v, ts = T.step(p, v, mass, ts)
                Pl.append(p)
            P = torch.stack(Pl).double()
            r["err"] = {str(k): pos_err(P, Pt, k) for k in (5, 10, 20)}
            r["diff_single"] = {str(k): float((P[k] - Ps[k]).norm(dim=-1).mean()) for k in (5, 10, 20)}
            r["dP"] = float((v.sum(0).norm() - vel.sum(0).norm()).abs())
            r["stats"] = dict(T.stats)
            row[name] = r
        rows.append(row)
        print("big", seed, {k: (v["err"]["20"], v["diff_single"]["20"]) for k, v in row.items() if isinstance(v, dict) and "err" in v}, row["single_err"], flush=True)
    json.dump(rows, open(out, "w"))


@torch.no_grad()
def phase_snaps(model, dev, out, steps=100):
    """Long rollouts (warm1200 and cold300, seed 4738) saved for the frame review; err@50/100 vs truth and vs single recorded."""
    keep = (0, 5, 10, 20, 50, 100)
    names = ["naive", "g16_fx", "g16_fx_far", "t32_g16_fx_far"]
    summ = {}
    for fam in ("warm1200", "cold300"):
        n, sigma, vfac = FAMILIES[fam]
        p0, v0 = make_scene(n, sigma, vfac, 128.0, 4738, 0)
        Pt, _ = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), steps, DT, SUB, EPS)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)
        mass = torch.ones(n, device=dev)
        single_setup(model)
        msk = torch.ones(1, n, device=dev)
        st = (model.init_field(1, dev)[0], None)
        p, v = pos[None], vel[None]
        runs = {}
        Pl = [p[0]]
        for _ in range(steps):
            p, v, st, _ = model.step(p, v, mass[None], msk, st)
            Pl.append(p[0])
        runs["single"] = torch.stack(Pl)
        for name in names:
            T = make_tiled(model, VARIANTS[name])
            ts = (T.init_field(dev), None)
            p, v = pos, vel
            Pl = [p]
            for _ in range(steps):
                p, v, ts = T.step(p, v, mass, ts)
                Pl.append(p)
            runs[name] = torch.stack(Pl)
        single_setup(model)
        allr = {"exact": Pt.float(), **runs}
        for name, P in allr.items():
            np.savez(out.replace(".json", f"_{fam}_{name}.npz"), pos=P[list(keep)].cpu().numpy(), steps=np.array(keep))
        summ[fam] = {name: {"err": {str(k): pos_err(P, Pt, k) for k in (20, 50, 100)},
                            "diff_single": {str(k): float((P[k].double() - runs["single"][k].double()).norm(dim=-1).mean()) for k in (20, 50, 100)}}
                     for name, P in runs.items()}
        print("snaps", fam, summ[fam], flush=True)
    json.dump(summ, open(out, "w"))


def tmed(fn, reps=15, warm=3):
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


@torch.no_grad()
def phase_scale(model, dev, out):
    res = []
    for nt in (1, 2, 4):
        L = 64.0 * nt
        n = int(300 * nt ** 2)
        sigma = L / 8
        p0, v0 = make_scene(n, sigma, 1.0, L, 4738, 0)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)
        mass = torch.ones(n, device=dev)
        msk = torch.ones(1, n, device=dev)
        r = {"tiles": nt ** 2, "n": n, "extent": L}
        ops.retarget(model, round(L / 0.5), L)
        st0 = (model.init_field(1, dev)[0], None)
        st = [st0]

        def sstep():
            st[0] = model.step(pos[None], vel[None], mass[None], msk, st[0])[2]
        sstep()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        r["single_ms"] = tmed(sstep)
        r["single_peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        r["single_grid"] = round(L / 0.5)
        for tag, spec in (("ghost_fx", (nt, 64, 16, True, True, False)), ("ghost_fx_far", (nt, 64, 16, True, True, True))):
            if nt == 1:
                spec = (1, 64, 0, False, False, False)
            T = make_tiled(model, spec)
            ts = [(T.init_field(dev), None)]

            def tstep():
                ts[0] = T.step(pos, vel, mass, ts[0])[2]
            tstep()
            r[tag + "_batched_ms"] = tmed(tstep, reps=10)
            r[tag + "_stats"] = dict(T.stats)
            r[tag + "_G"] = T.G
            if not spec[5]:
                k = (nt ** 2) // 2 + nt // 2 if nt > 1 else 0
                fld = T.init_field(dev)
                T.force(pos, vel, mass, fld, tiles=[k])
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                r[tag + "_solo_ms_per_force"] = tmed(lambda: T.force(pos, vel, mass, fld, tiles=[k]), reps=15)
                r[tag + "_solo_peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
                r[tag + "_solo_stats"] = dict(T.stats)
                cells = T.G ** 2
                r[tag + "_halo_cells_per_tile"] = int(cells - T.gt ** 2)
                r[tag + "_ghost_bodies_per_tile"] = T.stats["ghosts_mean"]
        r["coarse_cells_total"] = int((round(L / 0.5) // 4) ** 2)
        res.append(r)
        print(r, flush=True)
    json.dump(res, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phase", choices=["force", "roll", "scale", "big", "snaps"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="naive,g16,g16_fx,g16_fx_far,g8_fx,g8_fx_far,g32_fx,t32_naive,t32_g16_fx,t32_g16_fx_far")
    ap.add_argument("--kernel", default="old", help="scatter_far kernel spec (old = shipped kmlp)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    model = load(a.ckpt, dev)
    if a.kernel != "old":
        from scripts import scatter_far
        scatter_far.install_spec(model, a.kernel)
    if a.phase == "force":
        phase_force(model, dev, a.out)
    elif a.phase == "roll":
        phase_roll(model, dev, a.out, a.variants.split(","))
    elif a.phase == "big":
        phase_big(model, dev, a.out)
    elif a.phase == "snaps":
        phase_snaps(model, dev, a.out)
    else:
        phase_scale(model, dev, a.out)


if __name__ == "__main__":
    main()
