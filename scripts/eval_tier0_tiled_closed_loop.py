"""Held-out scoring of a tier 0 in the tiled hierarchy (exact-field arm, halo exchange as trained): err@k and dE on seeds 9100/9200/9300
(24 scenes x 100 ticks, scripts/exp12_hier.eval_generic), the same checkpoint without the exchange (diagnostic), and the plain single-window
eval (scripts/train_tier0_ext.plain_eval, seeds 9000/12000).
Usage: PYTHONPATH=. python scripts/eval_tier0_tiled_closed_loop.py --ckpt <ckpt> --exchange halo --out <json>"""
import argparse
import json
import os

import torch

from scripts import exp12_hier as h
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf
from scripts import train_tier0_ext as tte
from scripts import train_tiered as ttr
from scripts import tier0_tiled_closed_loop as tc

SEEDS = (9100, 9200, 9300)


def load(ckpt, dev):
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    model = tte.build(argparse.Namespace(t0=st["t0"], dt=0.1), dev)
    model.load_state_dict(st["model"])
    return model.eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exchange", choices=["halo", "zero", "none"], default="halo")
    ap.add_argument("--eval-scenes", type=int, default=24)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda s: print(s, flush=True)
    model = load(a.ckpt, dev)
    res = {"tiled": {}, "tiled_no_exchange": {}, "plain": {}}
    for name, ex in (("tiled", a.exchange), ("tiled_no_exchange", "none")):
        if name == "tiled_no_exchange" and a.exchange == "none":
            continue
        tt = tc.TiledTier0(model, dev, chunk=a.chunk, exchange=ex)
        for sd in SEEDS:
            d = ttr.make_scenes(a.eval_scenes, a.eval_steps, sd, dev)
            res[name][str(sd)] = h.eval_generic(lambda p, v, m, k, st: tt.force(p, v, m, k, st), tt.init_state, d)
            log(f"{name} {sd} {json.dumps({k: round(v, 5) for k, v in res[name][str(sd)].items()})}")
    model.t0.ext_grid = model.t0.ext_poly = model.t0.ext_high = None
    model.t0.set_material(None)
    with torch.no_grad():
        res["plain"] = tte.plain_eval(model, argparse.Namespace(dt=0.1, eps=0.5), dev, log, "plain")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
