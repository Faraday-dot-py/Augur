"""B1 untiled vs tiled gravity, with the per-particle head zeroed, vs A1 and Exp E. Plummer N=300 at span 64 (one window, untiled) and
span 128 (2x2 tiles of 32+2x16). Prints err@20/100 and relative energy drift at ticks 20/100.
Usage: PYTHONPATH=. python scripts/tier0_b1_ablate.py --b1 armB1.pt --a1 armA1.pt --e E.pt --out b1_ablate.json"""
import argparse
import copy
import json

import torch

from scripts import tiered_adapters as ta
from scripts import tiered_eval as te


def zero_head(ad):
    ad = copy.deepcopy(ad)
    for p in ad.m.t0.headnet.parameters():
        p.data.zero_()
    ad.name += "-nohead"
    return ad


def run(model, span, dev, seed, steps):
    b = te.grav_batch("plummer", 300, span, seed, 2, dev)
    tl = te.wrap(model, span)
    rt = te.truth_roll(b, model.dt, steps)
    rm = te.model_roll(tl, b, steps)
    mk = b["mask"] > 0
    phys = b["phys"]
    out = dict(tiled=type(tl).__name__ == "Tiled", nonfinite=rm["first_nonfinite"].tolist())
    d = (rm["pos"] - rt["pos"]).norm(dim=-1)
    for k in (20, steps):
        out[f"err@{k}"] = float((d[:, k] * mk).sum() / mk.sum())
    e0 = phys.energy(rt["pos"][:, 0], rt["vel"][:, 0], b["mass"], b["mask"])
    for k in (20, steps):
        em = phys.energy(rm["pos"][:, k], rm["vel"][:, k], b["mass"], b["mask"])
        et = phys.energy(rt["pos"][:, k], rt["vel"][:, k], b["mass"], b["mask"])
        out[f"dE@{k}_model"] = float(((em - e0).abs() / e0.abs()).mean())
        out[f"dE@{k}_truth"] = float(((et - e0).abs() / e0.abs()).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--e", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", default="b1_ablate.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    b1 = ta.load_t0ext(a.b1, dev)
    models = {"B1": b1, "B1_nohead": zero_head(b1), "A1": ta.load_t0ext(a.a1, dev), "ExpE": te.load_model(f"scatter:{a.e}", dev, 0.1)}
    res = {}
    for nm, m in models.items():
        for span in (64.0, 128.0):
            res[f"{nm}@{int(span)}"] = r = run(m, span, dev, a.seed, a.steps)
            print(nm, int(span), json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
