"""Grid-edge force probe for the scatter-field model (square-wall artifact, mechanism 5).

Static: massless tracers on the +-x/+-y axes from 16 to 46 (step 0.25) in a clipped-Gaussian N-body scene, model acceleration vs exact fp64. Band = |coord| in
[edge-3, edge+1]. Reports max|a| model/exact in the band, sign-match fraction and sign changes inside the grid part of the band (edge-3, edge).
Rollout: N-body cluster plus massless tracers launched outward at 0.95c from r 22-26, relativistic kick-drift-kick, model vs exact fp64 truth. Reports tracers
at r>70 (per 400) and the reversal fraction of outgoing tracers (axial |v|>1 before and after, outward then inward) at max-norm coordinate 28-32.

Usage: python scripts/edge_probe.py --ckpt X.pt --variants zero,kext,... --scene-seeds 9000,12000 --out results/scatter_edge/probe_X.json
A variant is a comma list of tokens joined by '+': zero | kext | krep | nrep | unetrep | far0.05 | far0.25 | nomesh (see apply_variant).
"""
import argparse
import json
import math

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

C, DT, EPS = 10.0, 0.1, 0.5
XS = np.arange(16.0, 46.01, 0.25)


def load(ckpt, dev, variant="ms_kp_pot_v_g128", extra=None):
    model = tsf.build(tsf.EXPS["E"], variant, DT).to(dev)
    st = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
    if extra:
        extra(model)
    model.load_state_dict(st)
    model.eval()
    ops.chunked_pp(model)
    return model


def apply_variant(model, spec):
    model.kpad, model.npad, model.far = "zero", "zero", 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.padding_mode = "zeros"
    for tok in spec.split("+"):
        if tok in ("zero", ""):
            continue
        if tok == "kext":
            model.kpad = "ext"
        elif tok == "krep":
            model.kpad = "rep"
        elif tok == "nrep":
            model.npad = "rep"
        elif tok == "unetrep":
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.padding_mode = "replicate"
        elif tok.startswith("far"):
            model.far, model.far_band = 1, float(tok[3:])
        else:
            raise ValueError(tok)


def scene(n, seed):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0)


@torch.no_grad()
def static_probe(model, dev, n, seed, exact_cache):
    p0 = scene(n, seed)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    tp = np.concatenate([np.stack([XS * dx, XS * dy], 1) for dx, dy in dirs])
    pos = np.concatenate([p0, tp])
    mass = np.concatenate([np.ones(n), np.zeros(len(tp))])
    key = (n, seed)
    if key not in exact_cache:
        exact_cache[key] = ops.exact_acc(torch.tensor(pos, dtype=torch.float64, device=dev), EPS, torch.tensor(mass, dtype=torch.float64, device=dev)).cpu().numpy()[n:]
    ex = exact_cache[key]
    P = torch.tensor(pos, dtype=torch.float32, device=dev)[None]
    M = torch.tensor(mass, dtype=torch.float32, device=dev)[None]
    mk = torch.ones_like(M)
    a, _ = model.force(P, torch.zeros_like(P), M, mk, model.init_field(1, dev)[0])
    am = a[0, n:].cpu().numpy()
    L = len(XS)
    edge = model.extent / 2
    out = {"n": n, "seed": seed}
    mx_m, mx_e, sm, sc, cnt = [], [], 0, 0, 0
    prof = {}
    for i, (dx, dy) in enumerate(dirs):
        sl = slice(i * L, (i + 1) * L)
        ax = 0 if dx else 1
        sg = dx + dy
        rad_m = am[sl, ax] * sg
        rad_e = ex[sl, ax] * sg
        band = (XS >= edge - 3) & (XS <= edge + 1)
        inn = (XS >= edge - 3) & (XS < edge)
        mx_m.append(np.abs(rad_m[band]).max())
        mx_e.append(np.abs(rad_e[band]).max())
        sm += int((np.sign(rad_m[inn]) == np.sign(rad_e[inn])).sum())
        cnt += int(inn.sum())
        sc += int((np.diff(np.sign(rad_m[inn])) != 0).sum())
        prof[f"d{i}"] = {"model": rad_m.tolist(), "exact": rad_e.tolist()}
    out.update(band_max_model=float(np.mean(mx_m)), band_max_exact=float(np.mean(mx_e)), band_ratio=float(np.mean(mx_m) / np.mean(mx_e)),
               sign_match=sm / cnt, sign_changes_per_dir=sc / 4, interior_rel_err=float(np.mean(
                   [np.abs(am[i * L:(i + 1) * L, 0 if dirs[i][0] else 1] * (dirs[i][0] + dirs[i][1]) - ex[i * L:(i + 1) * L, 0 if dirs[i][0] else 1] * (dirs[i][0] + dirs[i][1]))[(XS >= 18) & (XS <= 27)].mean()
                    / np.abs(ex[i * L:(i + 1) * L, 0 if dirs[i][0] else 1])[(XS >= 18) & (XS <= 27)].mean() for i in range(4)])))
    out["profile"] = prof
    out["xs"] = XS.tolist()
    return out


def mom_of(v):
    vm = v.norm(dim=-1, keepdim=True).clamp(max=0.99 * C)
    return v * (1 - (vm / C) ** 2).clamp(min=1e-6).rsqrt()


@torch.no_grad()
def truth_rollout(pos0, vel0, mass, steps, dev):
    pos = torch.tensor(pos0, dtype=torch.float64, device=dev)
    mom = mom_of(torch.tensor(vel0, dtype=torch.float64, device=dev))
    Md = torch.tensor(mass, dtype=torch.float64, device=dev)
    traj = [pos.cpu().numpy()]
    h = DT / 4
    a = ops.exact_acc(pos, EPS, Md)
    for _ in range(steps):
        for _ in range(4):
            mom = mom + 0.5 * h * a
            pos = pos + h * sb.speed(mom, C)
            a = ops.exact_acc(pos, EPS, Md)
            mom = mom + 0.5 * h * a
        traj.append(pos.cpu().numpy())
    return np.stack(traj)


@torch.no_grad()
def model_rollout(model, pos0, vel0, mass, steps, dev):
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    mom = mom_of(torch.tensor(vel0, dtype=torch.float32, device=dev)[None])
    M = torch.tensor(mass, dtype=torch.float32, device=dev)[None]
    mk = torch.ones_like(M)
    f = model.init_field(1, dev)[0]
    traj = [pos[0].cpu().numpy()]
    a, f = model.force(pos, sb.speed(mom, C), M, mk, f)
    for _ in range(steps):
        mom = mom + 0.5 * DT * a
        pos = pos + DT * sb.speed(mom, C)
        a, f = model.force(pos, sb.speed(mom, C), M, mk, f)
        mom = mom + 0.5 * DT * a
        traj.append(pos[0].cpu().numpy())
    return np.stack(traj)


def tracer_metrics(traj, n, nt, lo=28.0, hi=32.0):
    tr = traj[:, n:]
    rmax_end = np.linalg.norm(tr[-1], axis=1)
    coord = np.abs(tr).max(-1)
    ax = np.abs(tr).argmax(-1)
    v = np.diff(tr, axis=0) / DT
    vax = np.take_along_axis(v, ax[:-1, :, None], 2)[..., 0] * np.sign(np.take_along_axis(tr[:-1], ax[:-1, :, None], 2)[..., 0])
    out_s = rev = 0
    for t in range(1, len(vax)):
        sel = (coord[t] >= lo) & (coord[t] < hi) & (vax[t - 1] > 1.0)
        out_s += int(sel.sum())
        rev += int((sel & (vax[t] < -1.0)).sum())
    return {"beyond70_per400": float((rmax_end > 70).sum() * 400 / nt), "beyond70": int((rmax_end > 70).sum()), "reversal_frac": rev / max(out_s, 1),
            "outgoing_samples": out_s, "median_r": float(np.median(rmax_end)), "p99_r": float(np.percentile(rmax_end, 99))}


def rollout_probe(model, dev, n, nt, steps, seed, truth_cache):
    p0 = scene(n, seed)
    rng = np.random.default_rng(seed + 1)
    v0 = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    v0 -= v0.mean(0)
    ang = rng.uniform(0, 2 * np.pi, nt)
    r0 = rng.uniform(22, 26, nt)
    tp = np.stack([r0 * np.cos(ang), r0 * np.sin(ang)], 1)
    tv = 0.95 * C * np.stack([np.cos(ang), np.sin(ang)], 1)
    pos0, vel0 = np.concatenate([p0, tp]), np.concatenate([v0, tv])
    mass = np.concatenate([np.ones(n), np.zeros(nt)])
    key = (n, nt, steps, seed)
    if key not in truth_cache:
        truth_cache[key] = truth_rollout(pos0, vel0, mass, steps, dev)
    tm = tracer_metrics(model_rollout(model, pos0, vel0, mass, steps, dev), n, nt)
    tt = tracer_metrics(truth_cache[key], n, nt)
    return {"model": tm, "truth": tt, "seed": seed, "n": n, "nt": nt, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variants", default="zero")
    ap.add_argument("--scene-seeds", default="9000,12000")
    ap.add_argument("--n-static", type=int, default=100000)
    ap.add_argument("--n-roll", type=int, default=20000)
    ap.add_argument("--nt", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = load(args.ckpt, dev)
    seeds = [int(s) for s in args.scene_seeds.split(",")]
    ec, tc, res = {}, {}, {}
    for var in args.variants.split(","):
        apply_variant(model, var)
        res[var] = {"static": [static_probe(model, dev, args.n_static, s, ec) for s in seeds],
                    "rollout": [rollout_probe(model, dev, args.n_roll, args.nt, args.steps, s, tc) for s in seeds]}
        for st in res[var]["static"]:
            print(f"VAR {var} static seed {st['seed']} band max model {st['band_max_model']:.0f} exact {st['band_max_exact']:.0f} ratio {st['band_ratio']:.1f} "
                  f"signmatch {st['sign_match']:.2f} changes {st['sign_changes_per_dir']:.1f} interior_err {st['interior_rel_err']:.3f}", flush=True)
        for ro in res[var]["rollout"]:
            print(f"VAR {var} roll seed {ro['seed']} model beyond70/400 {ro['model']['beyond70_per400']:.1f} rev {ro['model']['reversal_frac']:.4f} | truth {ro['truth']['beyond70_per400']:.1f} rev {ro['truth']['reversal_frac']:.4f}", flush=True)
    json.dump(res, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
