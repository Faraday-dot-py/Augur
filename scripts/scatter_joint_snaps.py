"""1000-tick snapshots + conserved-quantity series of several scatter checkpoints on one scene per regime (nr100, bh300; seed 4738), with exact-sim truth.
Usage (GPU): PYTHONPATH=. python scripts/scatter_joint_snaps.py --models name=ckpt:variant [...] --out results/scatter_joint/snaps
Writes <out>_<regime>.npz: truth_pos (T/10+1,N,2), pos_<name>, series_<name>_{E,vir,bound,rhalf,outgrid} (T+1), truth_series_*."""
import argparse

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_conservation as sc
from scripts import train_scatter_field as tsf

STEPS = 1000
SERIES = ("E", "vir", "bound", "rhalf", "outgrid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regimes", default="nr100,bh300")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()
    dev = torch.device("cuda")
    for reg in args.regimes.split(","):
        kind, n, per = sc.REGIMES[reg]
        pos0, vel0, seeds = sc.make_scenes(kind, n, per)
        i = int(np.where(seeds == 4738)[0][0])
        P0 = torch.tensor(pos0[i:i + 1], device=dev)
        V0 = torch.tensor(vel0[i:i + 1], device=dev)
        rel = kind == "bh"
        r0 = (P0 - P0.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()
        out = {}

        def record(tag, runner):
            rec = sc.Recorder(rel, args.eps, 64.0, r0, args.steps, 1, dev, args.dt)
            runner(rec)
            out[f"pos_{tag}"] = torch.stack(rec.snaps)[:, 0].numpy()
            for k in SERIES:
                out[f"series_{tag}_{k}"] = rec.rows[k][:, 0].float().cpu().numpy()

        record("truth", lambda rec: sc.roll_truth(P0, V0, rel, args.steps, 4, args.eps, rec, [0], args.dt))
        for spec in args.models:
            name, rest = spec.split("=", 1)
            ck, var = rest.split(":")
            model = tsf.build(tsf.EXPS["E"], var, args.dt).to(dev)
            sf.load_compat(model, torch.load(ck, map_location=dev, weights_only=False)["model"])
            model.eval()
            record(name, lambda rec: sc.roll_model(model, P0, V0, rel, args.steps, rec, [0], args.dt))
            print(reg, name, "done", flush=True)
        np.savez(f"{args.out}_{reg}.npz", **out)


if __name__ == "__main__":
    main()
