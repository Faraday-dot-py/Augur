"""Inference-only divergence diagnosis of three-dimensional-sparse-voxel-scatter-field checkpoints (10k..30k its). Needs model/sparse_voxel_scatter_field.py and
scripts/three_dimensional_sparse_voxel_scatter_field.py from the extension branch tarball on PYTHONPATH.
Modes: weights (CPU: per-layer weight stats, learned kernel scans), sweep (rollouts by checkpoint, N, family), components (tick-0 force parts, teacher-forced vs rollout, ablations), gap (train vs val err@20).
Usage: PYTHONPATH=<tarball dir> python scripts/voxel_divergence_diagnosis.py --mode all --ckpt-dir <dir> --out <dir> --device cuda
"""
import argparse
import json
import os
import signal

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import three_dimensional_sparse_voxel_scatter_field as t
from model import sparse_voxel_scatter_field as sv

CKPTS = {"10k": "voxel-it10000.pt", "15k": "voxel-it15000.pt", "20k": "voxel-it20000.pt", "25k": "voxel-it25000.pt", "30k": "voxel-at-30000-its.pt"}
FAMS = {"cold": ("plummer", 8.0, 0.3), "virial": ("plummer", 8.0, 1.0), "train-cold-ball": ("ball", 7.5, 0.35)}
NS = (50, 100, 200, 300, 500, 1000)
CNS = (200, 1000)
REPS = 4
BIG = [(4738, r) for r in range(12)] + [(sd, r) for sd in (9100, 9200, 9300) for r in range(4)]
TICKS = 20
BAD = 1e3
STOP = {"flag": False}


def log(s):
    print(s, flush=True)


def on_term(signum, frame):
    STOP["flag"] = True


def load(path, dev):
    m = t.build(dev)
    st = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(st["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval(), st["it"]


def scene_list(n):
    return BIG if n == 1000 else [(4738, r) for r in range(REPS)]


def scene(fam, n, rep, seed=4738):
    kind, a, q = FAMS[fam]
    rng = np.random.default_rng([seed, n, rep])
    return t.plummer(n, a, q, rng) if kind == "plummer" else t.cold_ball(n, a, q, rng)


def parts_acc(m, pos, vel, mass, mask):
    levels, u0, frac, bidx, real = m.structure(pos, mask)
    x0 = m.deposit(levels, u0, frac, bidx, real, vel, mass)
    top = len(levels) - 1
    mass_l = [x0[:, 0] * m.h ** 3]
    for l in range(top):
        mass_l.append(torch.zeros(levels[l].nparent, device=pos.device, dtype=pos.dtype).index_add(0, levels[l].parent, mass_l[l]))
    ak = m.coarse_kernel(levels[top], top, mass_l[top])
    n0 = levels[0].keys.shape[0]
    h = m.inp(torch.cat([torch.log1p(x0[:, :1].clamp_min(0)), x0[:, 1:] / (1 + x0[:, :1].clamp_min(0)), x0[:, :1].new_zeros(n0, 3)], -1))
    feats = []
    for l, L in enumerate(levels):
        h = h + m.lvl(h.new_full((1, 1), float(l) / 8))
        h = m.down[min(l, sv.SHARED_FROM)](h, L.nb)
        feats.append(h)
        if l < top:
            cnt = torch.zeros(L.nparent, 1, device=h.device, dtype=h.dtype).index_add(0, L.parent, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype))
            h = torch.zeros(L.nparent, h.shape[1], device=h.device, dtype=h.dtype).index_add(0, L.parent, h) / cnt
    akn = ak / (1 + ak.norm(dim=-1, keepdim=True))
    cur = feats[top] + m.inp(torch.cat([torch.zeros(feats[top].shape[0], 4, device=h.device, dtype=h.dtype), akn], -1))
    for l in range(top - 1, -1, -1):
        cur = m.up[min(l, sv.SHARED_FROM)](torch.cat([feats[l], cur[levels[l].parent]], -1), levels[l].nb)
    phi = m.out(cur)
    nb = levels[0].nb
    pp_ = torch.cat([phi, phi.new_zeros(1, 1)], 0)
    idx = {d: i for i, d in enumerate(sv.DELTAS)}
    grad = torch.cat([(pp_[nb[:, idx[(1, 0, 0)]]] - pp_[nb[:, idx[(-1, 0, 0)]]]), (pp_[nb[:, idx[(0, 1, 0)]]] - pp_[nb[:, idx[(0, -1, 0)]]]),
                      (pp_[nb[:, idx[(0, 0, 1)]]] - pp_[nb[:, idx[(0, 0, -1)]]])], -1) / (2 * m.h)
    a_grid = m.gather_corners(-grad, levels, u0, frac, bidx, real)
    a_far = ak[m.top_parent(levels, u0, bidx, real)] * real[..., None]
    a_pair = m.pair_acc(pos, mass, mask)
    return a_grid, a_far, a_pair


def momfix(a, mass, mask):
    mm = (mass * mask)[..., None]
    return a - (mm * a).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)


def clip_norm(a, c):
    n = a.norm(dim=-1, keepdim=True)
    return a * torch.clamp(c / n.clamp_min(1e-12), max=1.0)


def install(m, w=(1.0, 1.0, 1.0), caps=None):
    """Replace m.force with a variant: component weights (grid, far, pair) and optional per-component norm caps. Diagnostic only."""
    def force(pos, vel, mass, mask, field):
        parts = parts_acc(m, pos, vel, mass, mask)
        if caps is not None:
            parts = tuple(clip_norm(p, c) if c is not None else p for p, c in zip(parts, caps))
        a = sum(wi * p for wi, p in zip(w, parts))
        if m.momfix:
            a = momfix(a, mass, mask)
        return a * mask[..., None], field
    m.force = force


def uninstall(m):
    m.__dict__.pop("force", None)


def tens(p, v, dev):
    return torch.tensor(p, dtype=torch.float32, device=dev)[None], torch.tensor(v, dtype=torch.float32, device=dev)[None]


def roll(m, p0, v0, truth, dev, steps=TICKS):
    P, V = tens(p0, v0, dev)
    M = torch.ones(1, P.shape[1], device=dev)
    field = m.init_field(1, dev)
    pm0 = float(P.abs().max())
    out = {"bad_tick": None, "bad_channel": None, "vmax": {}, "pmax": {}, "err20": None, "exception": None}
    for k in range(1, steps + 1):
        try:
            with torch.no_grad():
                P, V, _, field, _ = sf._step_m(m, P, V, M, M, field)
        except RuntimeError as e:
            out["bad_tick"], out["bad_channel"], out["exception"] = k, "key-range-or-index-error", str(e)[:80]
            return out
        fin = bool(torch.isfinite(P).all() and torch.isfinite(V).all())
        pm = float(P.abs().max()) if fin else float("nan")
        vm = float(V.abs().max()) if fin else float("nan")
        if k in (1, 2, 5, 10, 20) or not fin:
            out["vmax"][str(k)], out["pmax"][str(k)] = vm, pm
        if not fin or vm > BAD or pm > 10 * pm0 + BAD:
            out["bad_tick"] = k
            out["bad_channel"] = "nonfinite" if not fin else ("velocity" if vm > BAD else "position")
            return out
    out["err20"] = float((P[0].double() - truth[steps]).norm(dim=-1).mean())
    return out


def exact(p0, v0, dev, steps=TICKS):
    P, V = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), steps, t.DT, t.SUB, t.EPS)
    return P, V


def weights(args):
    dev = "cpu"
    res = {}
    base = None
    r = torch.logspace(-2.5, np.log10(2.0), 200)
    rf = torch.logspace(-1, 2.5, 200)
    for name, f in CKPTS.items():
        m, it = load(f"{args.ckpt_dir}/{f}", dev)
        sd = {k: v for k, v in m.state_dict().items()}
        if base is None:
            base = sd
        layers = {k: {"l2": float(v.norm()), "maxabs": float(v.abs().max()), "delta_l2_vs_10k": float((v - base[k]).norm()), "finite": bool(torch.isfinite(v).all())} for k, v in sd.items()}
        with torch.no_grad():
            gp = m.ppmlp(torch.stack([r, torch.log(r + 0.05)], -1))[..., 0]
            rr = torch.sqrt(rf ** 2 + 1e-8)
            gf = m.kmlp(torch.stack([rr, torch.log(rr + m.h * 2.0)], -1))[..., 0]
        res[name] = {"ckpt_it": it, "total_l2": float(torch.sqrt(sum((v.double() ** 2).sum() for v in sd.values()))), "layers": layers,
                     "pair_kernel_max_abs": float(gp.abs().max()), "pair_kernel_argmax_r": float(r[gp.abs().argmax()]), "far_kernel_max_abs": float(gf.abs().max()), "far_kernel_argmax_r": float(rf[gf.abs().argmax()]),
                     "pair_kernel_at_r": {f"{x:.3g}": float(gp[(r - x).abs().argmin()]) for x in (0.01, 0.1, 0.5, 1.0, 2.0)}, "far_kernel_at_r": {f"{x:.3g}": float(gf[(rf - x).abs().argmin()]) for x in (1, 10, 100, 300)}}
        log(f"weights {name} it {it} total_l2 {res[name]['total_l2']:.4g} pair_max {res[name]['pair_kernel_max_abs']:.4g} far_max {res[name]['far_kernel_max_abs']:.4g}")
    ref = res["10k"]
    for name in res:
        res[name]["pair_max_ratio_vs_10k"] = res[name]["pair_kernel_max_abs"] / max(ref["pair_kernel_max_abs"], 1e-12)
        res[name]["far_max_ratio_vs_10k"] = res[name]["far_kernel_max_abs"] / max(ref["far_kernel_max_abs"], 1e-12)
        log(f"weights {name} pair_max_ratio_vs_10k {res[name]['pair_max_ratio_vs_10k']:.4g} far_max_ratio_vs_10k {res[name]['far_max_ratio_vs_10k']:.4g}")
    json.dump(res, open(f"{args.out}/weights.json", "w"), indent=1)


def features(m, p0, dev):
    P = torch.tensor(p0, dtype=torch.float32, device=dev)[None]
    M = torch.ones(1, P.shape[1], device=dev)
    lv = m.structure(P, M)[0]
    r = np.linalg.norm(p0, axis=1)
    return {"rmax": float(r.max()), "r99": float(np.quantile(r, 0.99)), "n_levels": len(lv), "level0_voxels": int(lv[0].keys.shape[0]), "top_voxels": int(lv[-1].keys.shape[0])}


def sweep(args, dev):
    path = f"{args.out}/sweep.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name, f in CKPTS.items():
        m, it = load(f"{args.ckpt_dir}/{f}", dev)
        for n in NS:
            for fam in FAMS:
                for sd, rep in scene_list(n):
                    key = f"{name}|{n}|{fam}|{sd}|{rep}"
                    if key in res:
                        continue
                    if STOP["flag"]:
                        json.dump(res, open(path, "w"))
                        raise SystemExit(143)
                    p0, v0 = scene(fam, n, rep, sd)
                    Pt, _ = exact(p0, v0, dev)
                    cv = t.cv_err(p0, v0, Pt, 20, dev)
                    o = roll(m, p0, v0, Pt, dev)
                    o["cv20"], o["ratio"] = cv, (o["err20"] / cv if o["err20"] is not None else None)
                    o["ckpt_it"] = it
                    o.update(features(m, p0, dev))
                    res[key] = o
                    log(f"sweep {key} bad_tick {o['bad_tick']} {o['bad_channel']} err20 {o['err20']} ratio {o['ratio']}")
        json.dump(res, open(path, "w"))
    return res


def components(args, dev):
    path = f"{args.out}/components.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in ("10k", "20k", "25k", "30k"):
        if name not in CKPTS:
            continue
        m, it = load(f"{args.ckpt_dir}/{CKPTS[name]}", dev)
        for n in CNS:
            for fam in ("cold", "virial"):
                for sd, rep in scene_list(n):
                    key = f"{name}|{n}|{fam}|{sd}|{rep}"
                    if key in res:
                        continue
                    if STOP["flag"]:
                        json.dump(res, open(path, "w"))
                        raise SystemExit(143)
                    p0, v0 = scene(fam, n, rep, sd)
                    P, V = tens(p0, v0, dev)
                    M = torch.ones(1, n, device=dev)
                    with torch.no_grad():
                        ag, af, ap = parts_acc(m, P, V, M, M)
                        a_full = m.force(P, V, M, M, m.init_field(1, dev))[0]
                        a_sum = momfix(ag + af + ap, M, M)
                        a_ex = ops.exact_acc(torch.tensor(p0, device=dev), t.EPS).float()[None]
                    consistency = float((a_full - a_sum).abs().max())
                    am = a_sum[0].norm(dim=-1)
                    exm = a_ex[0].norm(dim=-1)
                    b = int(am.argmax())
                    cell = torch.floor(P[0] / m.pp).long()
                    _, inv, cnt = torch.unique(cell, dim=0, return_inverse=True, return_counts=True)
                    occ = cnt[inv]
                    d = (P[0][:, None] - P[0][None]).norm(dim=-1) + torch.eye(n, device=dev) * 1e9
                    nn_d = d.min(1).values
                    relerr = (a_sum[0] - a_ex[0]).norm(dim=-1) / exm.pow(2).mean().sqrt()
                    bins = {}
                    for lab, sel in (("cell_count_le8", occ <= 8), ("cell_count_9_32", (occ > 8) & (occ <= 32)), ("cell_count_gt32_overflow", occ > 32)):
                        bins[lab] = {"bodies": int(sel.sum()), "rms_err_over_exact_rms": float(relerr[sel].pow(2).mean().sqrt()) if sel.any() else None, "max_total_acc": float(am[sel].max()) if sel.any() else None,
                                     "max_pair_acc": float(ap[0].norm(dim=-1)[sel].max()) if sel.any() else None}
                    cell_over = int((cnt > m.cell_cap).sum())
                    o = {"ckpt_it": it, "force_parts_consistency_max_abs": consistency, "exact_max_acc": float(exm.max()), "exact_rms_acc": float(exm.pow(2).mean().sqrt()),
                         "max_acc": {"grid": float(ag[0].norm(dim=-1).max()), "far": float(af[0].norm(dim=-1).max()), "pair": float(ap[0].norm(dim=-1).max()), "total": float(am.max())},
                         "ratio_to_exact_max": {k: float(v) / float(exm.max()) for k, v in (("grid", ag[0].norm(dim=-1).max()), ("far", af[0].norm(dim=-1).max()), ("pair", ap[0].norm(dim=-1).max()), ("total", am.max()))},
                         "rms_err_over_exact_rms": float(relerr.pow(2).mean().sqrt()), "median_err_over_exact_rms": float(relerr.median()), "argmax_body": {"radius": float(P[0][b].norm()), "cell_count": int(occ[b]), "nearest_neighbour": float(nn_d[b]),
                         "grid": float(ag[0][b].norm()), "far": float(af[0][b].norm()), "pair": float(ap[0][b].norm()), "exact": float(exm[b])}, "occupancy_bins": bins, "cells_over_cap": cell_over}
                    Pt, Vt = exact(p0, v0, dev)
                    tf, ro = [], []
                    Pr, Vr = P.clone(), V.clone()
                    fld = m.init_field(1, dev)
                    for k in range(1, 6):
                        try:
                            with torch.no_grad():
                                pt1, vt1, _, _, _ = sf._step_m(m, Pt[k - 1].float()[None], Vt[k - 1].float()[None], M, M, m.init_field(1, dev))
                            tf.append({"pos_err": float((pt1[0].double() - Pt[k]).norm(dim=-1).mean()), "vel_max": float(vt1.abs().max())})
                        except RuntimeError as e:
                            tf.append({"exception": str(e)[:60]})
                        try:
                            with torch.no_grad():
                                Pr, Vr, _, fld, _ = sf._step_m(m, Pr, Vr, M, M, fld)
                                if k == 1 and bool(torch.isfinite(Pr).all() and torch.isfinite(Vr).all()):
                                    g1, f1, p1 = parts_acc(m, Pr, Vr, M, M)
                                    e1 = ops.exact_acc(Pr[0].double(), t.EPS).norm(dim=-1).max()
                                    o["tick1_state_max_acc_over_exact_max"] = {"grid": float(g1.norm(dim=-1).max() / e1), "far": float(f1.norm(dim=-1).max() / e1), "pair": float(p1.norm(dim=-1).max() / e1)}
                            ro.append({"pos_err": float((Pr[0].double() - Pt[k]).norm(dim=-1).mean()), "vel_max": float(Vr.abs().max())})
                        except RuntimeError as e:
                            ro.append({"exception": str(e)[:60]})
                            break
                    o["teacher_forced_ticks_1_to_5"], o["rollout_ticks_1_to_5"] = tf, ro
                    o.update(features(m, p0, dev))
                    if name in ("25k", "30k") and n == max(CNS):
                        cap = 10.0 * float(exm.max())
                        variants = {"baseline": dict(), "grid-zeroed": dict(w=(0, 1, 1)), "far-zeroed": dict(w=(1, 0, 1)), "pair-zeroed": dict(w=(1, 1, 0)), "grid-capped-10x-exact-max": dict(caps=(cap, None, None)),
                                    "far-capped-10x-exact-max": dict(caps=(None, cap, None)), "pair-capped-10x-exact-max": dict(caps=(None, None, cap)), "all-capped-10x-exact-max": dict(caps=(cap, cap, cap)), "cell-cap-256": dict(cell=256)}
                        ab = {"baseline": roll(m, p0, v0, Pt, dev)}
                        for vn, kw in variants.items():
                            if vn == "baseline" or ab["baseline"]["bad_tick"] is None:
                                continue
                            m.cell_cap = kw.get("cell", 32)
                            install(m, kw.get("w", (1.0, 1.0, 1.0)), kw.get("caps"))
                            ab[vn] = roll(m, p0, v0, Pt, dev)
                            uninstall(m)
                            m.cell_cap = 32
                        o["ablations"] = ab
                        log(f"components {key} ablation bad_tick " + " ".join(f"{k}={v['bad_tick']}" for k, v in ab.items()))
                    res[key] = o
                    log(f"components {key} ratio_to_exact_max {o['ratio_to_exact_max']} rms_err_over_exact_rms {o['rms_err_over_exact_rms']:.4g} cells_over_cap {cell_over} consistency {consistency:.3g}")
        json.dump(res, open(path, "w"))
    return res


def scene_err(m, sc, dev):
    try:
        return t.val_err(m, sc, dev), 0
    except RuntimeError:
        pass
    errs = []
    for i in range(sc[0].shape[0]):
        try:
            errs.append(t.val_err(m, tuple(x[i:i + 1] for x in sc), dev))
        except RuntimeError:
            pass
    nf = sc[0].shape[0] - len(errs)
    return (float(np.mean(errs)) if errs else float("nan")), nf


def gap(args, dev):
    path = f"{args.out}/gap.json"
    train = t.train_scenes(args.gap_scenes, 20, 4738, dev)
    val = t.train_scenes(48, 20, t.VAL_SEED, dev)
    res = {}
    for name, f in CKPTS.items():
        m, it = load(f"{args.ckpt_dir}/{f}", dev)
        (tr, trf), (va, vaf) = scene_err(m, train, dev), scene_err(m, val, dev)
        res[name] = {"ckpt_it": it, "train_err20": tr, "val_err20": va, "val_over_train": va / tr, "train_scenes": args.gap_scenes, "train_failed_scenes": trf, "val_failed_scenes": vaf}
        log(f"gap {name} it {it} train err@20 {tr:.5g} val err@20 {va:.5g} val/train {va / tr:.4g} failed_scenes train {trf} val {vaf}")
    json.dump(res, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weights", "sweep", "components", "gap", "all"], default="all")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gap-scenes", type=int, default=96)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global NS, REPS, CNS
    if args.smoke:
        NS, REPS, CNS = (50, 200), 1, (200,)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(4738)
    signal.signal(signal.SIGTERM, on_term)
    for f in CKPTS.values():
        if not os.path.isfile(f"{args.ckpt_dir}/{f}"):
            raise SystemExit(f"MISSING {args.ckpt_dir}/{f}")
    dev = torch.device(args.device)
    if args.mode in ("weights", "all"):
        weights(args)
    if args.mode in ("gap", "all"):
        gap(args, dev)
    if args.mode in ("components", "all"):
        components(args, dev)
    if args.mode in ("sweep", "all"):
        sweep(args, dev)
    log("DONE")


if __name__ == "__main__":
    main()
