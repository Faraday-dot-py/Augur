"""One-step 2-body force-accuracy probe for scatter-field checkpoints (exp A config).
Two unit-mass bodies at separation d, random orientation and sub-cell offset, held fixed for `warm` steps so the
recurrent field settles; reports relative error of body-0 acceleration vs the softened Newtonian truth."""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.train_scatter_field import EXPS, build

DISTS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0)


def probe(model, dev, n=256, warm=6, eps=0.5, seed=4738):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for d in DISTS:
        c = (torch.rand(n, 1, 2, generator=g) - 0.5) * 4.0
        th = torch.rand(n, generator=g) * 2 * np.pi
        u = torch.stack([th.cos(), th.sin()], -1)[:, None]
        pos = torch.cat([c - 0.5 * d * u, c + 0.5 * d * u], 1).to(dev)
        vel = torch.zeros_like(pos)
        mass = torch.ones(n, 2, device=dev)
        mask = torch.ones(n, 2, device=dev)
        field = model.init_field(n, dev)
        with torch.no_grad():
            for _ in range(warm):
                _, _, _, field, dv = sf._step_m(model, pos, vel, mass, mask, field)
        a = dv / model.dt
        r = pos[:, 1] - pos[:, 0]
        at = r * ((r ** 2).sum(-1, keepdim=True) + eps ** 2) ** -1.5
        err = (a[:, 0] - at).norm(dim=-1) / at.norm(dim=-1)
        ang = (a[:, 0] / a[:, 0].norm(dim=-1, keepdim=True) * at / at.norm(dim=-1, keepdim=True)).sum(-1).clamp(-1, 1).acos()
        out[str(d)] = {"rel_err_mean": float(err.mean()), "rel_err_med": float(err.median()), "angle_err_deg": float(ang.mean() * 180 / np.pi),
                       "true_mag": float(at.norm(dim=-1).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--exp", default="A")
    ap.add_argument("--ckpt-dir", default="checkpoints/scatter_field")
    ap.add_argument("--out", default="results/scatter_field/force_probe.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for name in args.variants.split(","):
        m = build(EXPS[args.exp], name, 0.1).to(dev)
        try:
            m.load_state_dict(torch.load(f"{args.ckpt_dir}/{args.exp}_{name}.pt", map_location=dev, weights_only=False)["model"])
        except RuntimeError:
            print(name, "checkpoint incompatible with current code, skipped", flush=True)
            continue
        res[name] = probe(m, dev)
        print(name, {d: round(v["rel_err_mean"], 3) for d, v in res[name].items()}, flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
