"""Mixed-law dec1-dec4 activation statistics by law and amplitude (curiosity, observations only, no pass/kill verdict).
Per (checkpoint, trained shape, amplitude): 24 single-structure scenes from sm.gen_cases (seed 4738), tick-0 forward from a zero field; per layer dec1-dec4 mean|act|, max|act|,
fraction exactly zero, plus the tick-0 force relative error against the fp64 exact force (same definition as scatter_multi_eval.eval_case). Per-cell resume.
Usage: PYTHONPATH=. python scripts/mixed_law_dec3_activation_by_law_and_amplitude.py --out results/mixed_law_dec3_activation
"""
import argparse
import json
import os
import signal
import sys

import torch

from scripts import scatter_multi as sm
from scripts.law_recovery_probe import load_multi

SEED = 4738
AMPS = (0.1, 0.316, 1.0, 3.16, 10.0)
LAYERS = ("dec1", "dec2", "dec3", "dec4")
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


@torch.no_grad()
def run_cell(model, dev, shape, amp, n, steps):
    d = sm.gen_cases([(shape, "single", amp)] * n, SEED, dev, steps)
    P, V, mask, cond, par = d["P"][:, 0], d["V"][:, 0], d["mask"], d["cond"], d["params"]
    S, N = mask.shape
    typ = torch.zeros(S, N, device=dev)
    mass = torch.ones(S, N, device=dev)
    model.set_law(cond, typ)
    store = {}
    hs = [model.net.dec[int(k[3:])].register_forward_hook(lambda m, i, o, k=k: store.__setitem__(k, o.detach())) for k in LAYERS]
    try:
        a, _ = model.force(P, V, mass, mask, model.init_field(S, dev)[0])
    finally:
        for h in hs:
            h.remove()
    at = sm.exact_acc(P.double(), typ, mask.double(), par).float()
    inside = (P.abs() < 32).all(-1).float() * mask
    rec = {"n_scenes": int(S), "force_rel": float(((a - at).norm(dim=-1) * inside).sum() / (at.norm(dim=-1) * inside).sum())}
    for k in LAYERS:
        x = store[k]
        rec[k] = {"mean_abs": float(x.abs().double().mean()), "max_abs": float(x.abs().max()), "frac_zero": float((x == 0).double().mean()), "numel": x.numel()}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    n = 2 if a.smoke else a.n
    path = os.path.join(a.out, "result.json")
    res = json.load(open(path)) if os.path.exists(path) else {"seed": SEED, "amps": list(AMPS), "n_scenes": n, "cells": {}}
    res.pop("done", None)
    signal.signal(signal.SIGTERM, on_term)
    specs = [("smoke", None)] if a.smoke else [tuple(s.split(":")) for s in a.models.split(",")]
    for name, ckpt in specs:
        if stop:
            break
        torch.manual_seed(SEED)
        model = smoke_model(dev) if a.smoke else load_multi(ckpt, dev)
        for shape in sm.TRAIN_SHAPES:
            for amp in AMPS:
                key = f"{name}|{shape}|{amp}"
                if key in res["cells"]:
                    print("skip", key, flush=True)
                    continue
                if stop:
                    break
                res["cells"][key] = run_cell(model, dev, shape, amp, n, a.steps)
                save(path, res)
                r = res["cells"][key]
                print(key, "force_rel", round(r["force_rel"], 4), "dec3 frac_zero", round(r["dec3"]["frac_zero"], 4), flush=True)
        del model
    if stop:
        print("SIGTERM: partial result saved", flush=True)
        sys.exit(143)
    expect = len(specs) * len(sm.TRAIN_SHAPES) * len(AMPS)
    if len(res["cells"]) != expect:
        print("INCOMPLETE", len(res["cells"]), expect, flush=True)
        sys.exit(1)
    res["done"] = True
    save(path, res)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
