"""Untiled (span 64) vs tiled (span 128) Plummer gravity for several tier-0 checkpoints, with an optional per-model head ablation.
Prints err@20/100, dE and the tiled/untiled err@20 ratio.
A path suffixed +matgate evaluates that checkpoint with the material gate switched on.
Usage: PYTHONPATH=. python scripts/tier0_tiling_cmp.py --models S1=a.pt,A1=b.pt --e E.pt --out tiling_cmp.json"""
import argparse
import json

import torch

from scripts import tiered_adapters as ta
from scripts import tiered_eval as te
from scripts.tier0_b1_ablate import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--e", default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="tiling_cmp.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    models = {}
    for kv in [x for x in a.models.split(",") if x]:
        k, _, v = kv.partition("=")
        gate = v.endswith("+matgate")
        models[k] = ta.load_t0ext(v[:-len("+matgate")] if gate else v, dev)
        if gate:
            models[k].m.t0.cfg.matgate = True
    if a.e:
        models["ExpE"] = te.load_model(f"scatter:{a.e}", dev, 0.1)
    res = {}
    for nm, m in models.items():
        for span in (64.0, 128.0):
            res[f"{nm}@{int(span)}"] = r = run(m, span, dev, a.seed, a.steps)
            print(nm, int(span), json.dumps(r), flush=True)
        res[f"{nm}_tiled_over_untiled_err20"] = res[f"{nm}@128"]["err@20"] / res[f"{nm}@64"]["err@20"]
        print(nm, "tiled/untiled err@20", res[f"{nm}_tiled_over_untiled_err20"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
