"""Interior force offset probe (audit row 10, docs/debugging/director-feasibility-2026-11-07.md).
Radial force ratio learned/exact on a uniform disk (radius 24, eps 0.5, G=1) as particle count N and per-body mass vary, to attribute the roughly 1.55x interior offset
seen at 1e6 bodies (experiment-log.md, 2026-10 edge-check entry) to mass density out of training range, the pair term, or the grid kernel (plain PM with the analytic kernel).
Truth = fp64 softened point sum (scatter_scaling_ops.exact_acc; targets subsampled above N=100000). Ratio = sum(a . rhat) / sum(a_exact . rhat) over interior bins r < 0.75 R.
Cells are written to --out after each one and skipped on rerun.
Usage: PYTHONPATH=. python scripts/interior_force_offset_probe.py --ckpt kern/E_frozen.pt --out results/interior_force_offset_probe.json
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.pm_baseline_force_error import EPS, learned_acc, pm_acc, sample

R = 24.0
BINS = (0.0, 0.25, 0.5, 0.75)
CELLS = (("fixed_body_mass", 1000, 1.0, 4), ("fixed_body_mass", 10000, 1.0, 3), ("fixed_body_mass", 30000, 1.0, 2), ("fixed_body_mass", 100000, 1.0, 2),
         ("fixed_total_mass", 10000, 0.1, 3), ("fixed_total_mass", 100000, 0.01, 2), ("fixed_body_mass", 1000000, 1.0, 1))
SUB = 8192


@torch.no_grad()
def exact_targets(pos, mass, idx, eps, elems=2.0e8):
    out = torch.empty(idx.numel(), 2, dtype=pos.dtype, device=pos.device)
    c = max(1, int(elems / pos.shape[0]))
    for s in range(0, idx.numel(), c):
        i = idx[s:s + c]
        d = pos[None, :, :] - pos[i, None, :]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
        inv[torch.arange(i.numel(), device=pos.device), i] = 0.0
        out[s:s + c] = (d * (inv * mass[None, :])[..., None]).sum(1)
    return out


def radial_ratios(a, at, pos, idx):
    p = pos[idx]
    r = p.norm(dim=-1)
    rh = p / r.clamp_min(1e-9)[:, None]
    ar = (a[idx] * rh).sum(-1)
    atr = (at * rh).sum(-1)
    am = a[idx].norm(dim=-1)
    atm = at.norm(dim=-1)
    out = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (r >= lo * R) & (r < hi * R)
        if int(m.sum()) < 20:
            continue
        out[f"{lo:.2f}-{hi:.2f}"] = {"n": int(m.sum()), "ratio_of_means": float(ar[m].sum() / atr[m].sum()), "median_norm_ratio": float((am[m] / atm[m]).median())}
    m = r < 0.75 * R
    out["interior"] = {"n": int(m.sum()), "ratio_of_means": float(ar[m].sum() / atr[m].sum()), "median_norm_ratio": float((am[m] / atm[m]).median())}
    return out


def build(dev, ckpt):
    model = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    if ckpt != "random":
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-n", type=int, default=1000000)
    ap.add_argument("--scene-cap", type=int, default=99)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = build(dev, a.ckpt)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"seed": a.seed, "warm": a.warm, "eps": EPS, "radius": R, "cells": {}}
    pp_orig = model.pp_acc
    for mode, n, m0, ns in CELLS:
        key = f"{mode}_n{n}"
        if n > a.max_n or key in res["cells"]:
            continue
        rng = np.random.default_rng(a.seed + n)
        ns = min(ns, a.scene_cap)
        scenes = []
        try:
            for s in range(ns):
                pos = torch.tensor(sample("uniform_disk", n, rng), dtype=torch.float32, device=dev)[None]
                mass = torch.full((1, n), m0, device=dev)
                mask = torch.ones(1, n, device=dev)
                vel = torch.zeros_like(pos)
                idx = torch.arange(n, device=dev) if n <= 100000 else torch.randperm(n, device=dev, generator=torch.Generator(device=dev).manual_seed(a.seed))[:SUB]
                at = exact_targets(pos[0].double(), mass[0].double(), idx, EPS).float()
                if n >= 10000:
                    model.opt.add("adapt")
                with torch.no_grad():
                    variants = {"learned": learned_acc(model, pos, vel, mass, mask, a.warm)[0], "plain_pm": pm_acc(model, pos, vel, mass, mask)[0]}
                    model.pp_acc = lambda *args, **kw: torch.zeros_like(pos)
                    variants["learned_no_pair_term"] = learned_acc(model, pos, vel, mass, mask, a.warm)[0]
                    model.pp_acc = pp_orig
                model.opt.discard("adapt")
                scenes.append({k: radial_ratios(v, at, pos[0], idx) for k, v in variants.items()})
                del pos, mass, mask, vel, at, variants
                torch.cuda.empty_cache() if dev.type == "cuda" else None
        except torch.cuda.OutOfMemoryError:
            model.pp_acc = pp_orig
            model.opt.discard("adapt")
            res["cells"][key] = {"oom": True, "scenes_done": len(scenes)}
            torch.cuda.empty_cache()
            json.dump(res, open(a.out, "w"), indent=1)
            print(key, "OOM", flush=True)
            continue
        row = {"mode": mode, "n": n, "body_mass": m0, "scenes": len(scenes), "per_scene": scenes}
        for v in scenes[0]:
            row[v] = {"interior_ratio_of_means": float(np.mean([s[v]["interior"]["ratio_of_means"] for s in scenes])),
                      "interior_median_norm_ratio": float(np.mean([s[v]["interior"]["median_norm_ratio"] for s in scenes])),
                      "bins": {b: float(np.mean([s[v][b]["ratio_of_means"] for s in scenes if b in s[v]])) for b in scenes[0][v] if b != "interior"}}
        res["cells"][key] = row
        json.dump(res, open(a.out, "w"), indent=1)
        print(key, {v: round(row[v]["interior_ratio_of_means"], 4) for v in scenes[0]}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
