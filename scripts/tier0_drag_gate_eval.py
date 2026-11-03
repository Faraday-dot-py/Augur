"""Drag-scene eval of an existing head+mat checkpoint with the material gate switched on (inference-only gating).
Usage: PYTHONPATH=. python scripts/tier0_drag_gate_eval.py --ckpt armB1.pt --baseline E.pt --out drag_gate.json [--gate]"""
import argparse
import json

import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m
from scripts import train_scatter_field as tsf
from scripts import train_tier0_ext as tte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--out", default="drag_gate.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", a.dt).to(dev)
    t0m.attach_tier0(model, ck["t0"])
    model.load_state_dict(ck["model"])
    model.eval()
    if a.gate:
        model.t0.cfg.matgate = True
    res = tte.drag_eval(model, a, dev, lambda s: print(s, flush=True))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
