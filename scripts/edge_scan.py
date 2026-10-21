"""Scan for edge-related artifacts other than the kernel zero-pad spike: (a) momfix beyond the grid and star+light, (d) recurrent field/a_prev at the edge,
(g) fp16 body positions at the edge. Usage: python scripts/edge_scan.py --ckpt E.pt --out scan.json"""
import argparse
import json

import numpy as np
import torch

from scripts import edge_probe as ep
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

DT, EPS = 0.1, 0.5


def force_at(model, pos, mass, dev):
    P = torch.tensor(pos, dtype=torch.float32, device=dev)[None]
    M = torch.tensor(mass, dtype=torch.float32, device=dev)[None]
    a, _ = model.force(P, torch.zeros_like(P), M, torch.ones_like(M), model.init_field(1, dev)[0])
    return a[0].cpu().numpy()


@torch.no_grad()
def scan_momfix(model, dev):
    out = {}
    n = 300
    rng = np.random.default_rng(9000)
    p0 = np.clip(rng.normal(0, 8.0, (n, 2)), -29, 29)
    xs = np.arange(34.0, 61.0, 2.0)
    tp = np.stack([xs, np.zeros_like(xs)], 1)
    pos = np.concatenate([p0, tp])
    mass = np.concatenate([np.ones(n), np.zeros(len(tp))])
    ex = ops.exact_acc(torch.tensor(pos, dtype=torch.float64, device=dev), EPS, torch.tensor(mass, dtype=torch.float64, device=dev)).cpu().numpy()[n:, 0]
    for mf in (True, False):
        model.momfix = mf
        a = force_at(model, pos, mass, dev)[n:, 0]
        out[f"outside_momfix_{mf}"] = {"model_mean": float(a.mean()), "exact_mean": float(ex.mean()), "model": a.tolist(), "exact": ex.tolist()}
    rng = np.random.default_rng(9000)
    errs = {True: [], False: []}
    selfs = {True: [], False: []}
    for s in range(24):
        M = rng.uniform(8, 30)
        r = rng.uniform(3, 28, 10)
        th = rng.uniform(0, 2 * np.pi, 10)
        pos = np.concatenate([[[0.0, 0.0]], np.stack([r * np.cos(th), r * np.sin(th)], 1)])
        mass = np.concatenate([[M], np.exp(rng.uniform(np.log(0.01), np.log(0.5), 10))])
        ex = ops.exact_acc(torch.tensor(pos, dtype=torch.float64, device=dev), EPS, torch.tensor(mass, dtype=torch.float64, device=dev)).cpu().numpy()
        for mf in (True, False):
            model.momfix = mf
            a = force_at(model, pos, mass, dev)
            errs[mf].append(float(np.linalg.norm(a[1:] - ex[1:], axis=1).mean() / np.linalg.norm(ex[1:], axis=1).mean()))
            selfs[mf].append(float(np.linalg.norm(a[0])))
    model.momfix = True
    out["star_light"] = {"light_rel_err_momfix": float(np.mean(errs[True])), "light_rel_err_nomomfix": float(np.mean(errs[False])),
                         "star_self_accel_momfix": float(np.mean(selfs[True])), "star_self_accel_nomomfix": float(np.mean(selfs[False]))}
    return out


@torch.no_grad()
def scan_field(model, dev):
    n, nt, steps, seed = 20000, 2000, 100, 9000
    p0 = ep.scene(n, seed)
    rng = np.random.default_rng(seed + 1)
    v0 = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    v0 -= v0.mean(0)
    ang = rng.uniform(0, 2 * np.pi, nt)
    r0 = rng.uniform(22, 26, nt)
    tp = np.stack([r0 * np.cos(ang), r0 * np.sin(ang)], 1)
    tv = 0.95 * ep.C * np.stack([np.cos(ang), np.sin(ang)], 1)
    pos = torch.tensor(np.concatenate([p0, tp]), dtype=torch.float32, device=dev)[None]
    mom = ep.mom_of(torch.tensor(np.concatenate([v0, tv]), dtype=torch.float32, device=dev)[None])
    M = torch.tensor(np.concatenate([np.ones(n), np.zeros(nt)]), dtype=torch.float32, device=dev)[None]
    mk = torch.ones_like(M)
    f = model.init_field(1, dev)[0]
    G = model.grid
    rows = []
    for t in range(steps + 1):
        a, f = model.force(pos, ep.sb.speed(mom, ep.C), M, mk, f)
        if t % 10 == 0:
            f0 = f[0, 0]
            ring = torch.cat([f0[:3].flatten(), f0[-3:].flatten(), f0[:, :3].flatten(), f0[:, -3:].flatten()]).abs().max()
            band = torch.cat([f0[4:8].flatten(), f0[-8:-4].flatten()]).abs().max()
            inter = f0[16:-16, 16:-16].abs().max()
            at = a[0, n:].norm(dim=-1)
            rows.append({"t": t, "field_ring3": float(ring), "field_band4_8": float(band), "field_interior": float(inter), "max_acc_tracer": float(at.max()), "max_acc_cluster": float(a[0, :n].norm(dim=-1).max())})
        mom = mom + 0.5 * DT * a
        pos = pos + DT * ep.sb.speed(mom, ep.C)
        a, f = model.force(pos, ep.sb.speed(mom, ep.C), M, mk, f)
        mom = mom + 0.5 * DT * a
    return rows


@torch.no_grad()
def scan_fp16(model, dev):
    n = 100000
    p0 = ep.scene(n, 9000)
    xs = np.arange(26.0, 34.01, 0.0625)
    tp = np.stack([xs, np.zeros_like(xs)], 1)
    mass = np.concatenate([np.ones(n), np.zeros(len(tp))])
    out = {}
    pos = np.concatenate([p0, tp])
    P32 = torch.tensor(pos, dtype=torch.float32, device=dev)
    P16 = P32.half().float()
    a32 = model.force(P32[None], torch.zeros_like(P32)[None], torch.tensor(mass, dtype=torch.float32, device=dev)[None], torch.ones(1, len(pos), device=dev), model.init_field(1, dev)[0])[0][0, n:, 0]
    a16 = model.force(P16[None], torch.zeros_like(P16)[None], torch.tensor(mass, dtype=torch.float32, device=dev)[None], torch.ones(1, len(pos), device=dev), model.init_field(1, dev)[0])[0][0, n:, 0]
    d = (a16 - a32).abs().cpu().numpy()
    band = (xs >= 29) & (xs <= 33)
    inn = xs < 28
    out = {"max_abs_diff_band": float(d[band].max()), "max_abs_diff_interior": float(d[inn].max()), "max_abs_a32_band": float(a32.abs().cpu().numpy()[band].max()),
           "position_quantum_16_32": 2 ** -6}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    for name, opts in (("base", "none"), ("kext", "kext")):
        model = ep.load(args.ckpt, dev)
        sf.apply_opts(model, opts)
        ops.chunked_pp(model)
        res[name] = {"momfix": scan_momfix(model, dev), "field": scan_field(model, dev), "fp16": scan_fp16(model, dev)}
        mf = res[name]["momfix"]
        print(f"SCAN {name} outside a mean model(momfix) {mf['outside_momfix_True']['model_mean']:.4f} model(no momfix) {mf['outside_momfix_False']['model_mean']:.4f} exact {mf['outside_momfix_True']['exact_mean']:.4f}", flush=True)
        print(f"SCAN {name} star+light {json.dumps(mf['star_light'])}", flush=True)
        for r in res[name]["field"]:
            print(f"SCAN {name} field {json.dumps(r)}", flush=True)
        print(f"SCAN {name} fp16 {json.dumps(res[name]['fp16'])}", flush=True)
    json.dump(res, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
