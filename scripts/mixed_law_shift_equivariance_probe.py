"""Shift-equivariance probe of the mixed-law scatter field (pass line: median relative force change above 2% on at least 3 of 6 laws; per-law decision requires x, y and diag on every checkpoint, per-cell scores also reported).
Per (checkpoint, trained shape): 32 single-structure amplitude-1 scenes (seed 4738), tick-0 force from a zero field for the scene and for the scene shifted rigidly by half a grid cell
(model.h / 2; x, y and diagonal). The force is translation-covariant so the shifted force is compared unchanged. Metric: median over bodies of |a_shift - a| / |a| for bodies with
|a| above the 10th percentile of their scene. A zero shift gives the determinism floor; the model error against the fp64 exact force uses the same metric for scale.
Usage: PYTHONPATH=. python scripts/mixed_law_shift_equivariance_probe.py --out results/mixed_law_shift_equivariance_probe
"""
import argparse
import json
import os
import signal
import sys

import numpy as np
import torch

from scripts import scatter_multi as sm
from scripts.law_recovery_probe import load_multi

SEED = 4738
PASS_MEDIAN = 0.02
PASS_LAWS = 3
MODELS = "main_s4738:ckpt/main_s4738.pt,main_s4739:ckpt/main_s4739.pt"

stop = False


def on_term(signum, frame):
    global stop
    stop = True


def smoke_model(dev):
    m = sm.build()
    torch.nn.init.normal_(m.net.out.weight, std=0.05)
    return m.to(dev).eval()


def save(path, res):
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(path + ".tmp", path)


def rel_bodies(a, ref, mask, valid):
    """per-scene list of |a - ref| / |ref| over valid bodies with |ref| above that scene's 10th percentile"""
    out = []
    for s in range(a.shape[0]):
        m = (mask[s] > 0) & valid[s]
        r = ref[s][m].norm(dim=-1)
        e = (a[s][m] - ref[s][m]).norm(dim=-1)
        keep = r > torch.quantile(r, 0.1)
        out.append((e[keep] / r[keep]).double().cpu().numpy())
    return out


@torch.no_grad()
def run_cell(model, dev, shape, n, steps):
    d = sm.gen_cases([(shape, "single", 1.0)] * n, SEED, dev, steps, tag=1)
    P, V, mask, cond, par = d["P"][:, 0], d["V"][:, 0], d["mask"], d["cond"], d["params"]
    S, N = mask.shape
    typ = torch.zeros(S, N, device=dev)
    mass = torch.ones(S, N, device=dev)
    model.set_law(cond, typ)
    half = float(model.h) / 2
    shifts = {"zero": (0.0, 0.0), "x": (half, 0.0), "y": (0.0, half), "diag": (half, half)}

    def force(pos):
        return model.force(pos, V, mass, mask, model.init_field(S, dev)[0])[0]

    base = force(P)
    at = sm.exact_acc(P.double(), typ, mask.double(), par).float()
    valid = ((P.abs() < 32 - 2 * half).all(-1))
    rec = {"n_scenes": int(S), "half_cell": half}
    for k, (dx, dy) in shifts.items():
        sh = torch.tensor([dx, dy], device=dev, dtype=P.dtype)
        v = valid & ((P + sh).abs() < 32).all(-1)
        per = rel_bodies(force(P + sh), base, mask, v)
        allb = np.concatenate(per)
        rec[k] = {"median": float(np.median(allb)), "mean": float(allb.mean()), "p90": float(np.percentile(allb, 90)), "median_of_scene_medians": float(np.median([np.median(p) for p in per])),
                  "n_bodies": int(len(allb))}
    allb = np.concatenate(rel_bodies(base, at, mask, valid))
    rec["model_vs_exact"] = {"median": float(np.median(allb)), "mean": float(allb.mean()), "p90": float(np.percentile(allb, 90)), "n_bodies": int(len(allb))}
    return rec


def score(res, names):
    out = {}
    for name in names:
        for k in ("x", "y", "diag"):
            above = [s for s in sm.TRAIN_SHAPES if res["cells"][f"{name}|{s}"][k]["median"] > PASS_MEDIAN]
            out[f"{name}|{k}"] = {"laws_above": above, "n_above": len(above), "pass": len(above) >= PASS_LAWS}
    strict = [s for s in sm.TRAIN_SHAPES if all(res["cells"][f"{name}|{s}"][k]["median"] > PASS_MEDIAN for name in names for k in ("x", "y", "diag"))]
    out["combined"] = {"rule": "law counts only if median > PASS_MEDIAN for x, y and diag on every checkpoint", "laws_above": strict, "n_above": len(strict),
                       "verdict": "PASS" if len(strict) >= PASS_LAWS else "KILL"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    n = 2 if a.smoke else a.n
    path = os.path.join(a.out, "result.json")
    res = json.load(open(path)) if os.path.exists(path) else {"seed": SEED, "n_scenes": n, "pass_line": f"median > {PASS_MEDIAN} on >= {PASS_LAWS} of 6 laws", "cells": {}}
    res.pop("done", None)
    signal.signal(signal.SIGTERM, on_term)
    specs = [("smoke", None)] if a.smoke else [tuple(s.split(":")) for s in a.models.split(",")]
    for name, ckpt in specs:
        if stop:
            break
        torch.manual_seed(SEED)
        model = smoke_model(dev) if a.smoke else load_multi(ckpt, dev)
        for shape in sm.TRAIN_SHAPES:
            key = f"{name}|{shape}"
            if key in res["cells"]:
                print("skip", key, flush=True)
                continue
            if stop:
                break
            res["cells"][key] = run_cell(model, dev, shape, n, a.steps)
            save(path, res)
            r = res["cells"][key]
            print(key, "zero", r["zero"]["median"], "x", round(r["x"]["median"], 5), "y", round(r["y"]["median"], 5), "diag", round(r["diag"]["median"], 5),
                  "vs exact", round(r["model_vs_exact"]["median"], 5), flush=True)
        del model
    if stop:
        print("SIGTERM: partial result saved", flush=True)
        sys.exit(143)
    expect = len(specs) * len(sm.TRAIN_SHAPES)
    if len(res["cells"]) != expect:
        print("INCOMPLETE", len(res["cells"]), expect, flush=True)
        sys.exit(1)
    res["score"] = score(res, [s[0] for s in specs])
    res["done"] = True
    save(path, res)
    print(json.dumps(res["score"]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
