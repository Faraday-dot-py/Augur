"""NVE energy-drift test for the scatter field on LJ MD (docs/debugging/scatter-lj.md, hole-1 section). From the first truth frame of each scene
(truth prefixes = different seeds), roll out --frames frames and report change in true-LJ E/N, final T and finite flag.
Usage: PYTHONPATH=. python -m scripts.scatter_lj_drift --ckpt ckpt/P_final.pt --tag P --prefixes results/lj,results/s4739/lj,results/s4740/lj"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_lj as sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prefixes", default="results/lj")
    ap.add_argument("--scenes", default="gas,liquid,solid,coexist")
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = sl.load_model(args.ckpt, dev)
    model.eval()
    out = {}
    for pre in args.prefixes.split(","):
        for name in args.scenes.split(","):
            z = np.load(f"{pre}_{name}.npz")
            nf = min(args.frames, len(z["pos"]) - 1)
            _, _, st, ok = sl.rollout(model, z["pos"][0].astype(np.float64), z["vel"][0].astype(np.float64), z["box"], nf, dev)
            n = len(st) - 1
            r = {"finite": bool(ok), "frames": n, "dE_per_N": float(st[-1, 3] - st[0, 3]), "T_end": float(st[-1, 1]),
                 "T_truth_end": float(z["stats"][min(n, len(z["stats"]) - 1), 1]),
                 "dE_truth": float(z["stats"][min(n, len(z["stats"]) - 1), 3] - z["stats"][0, 3])}
            out[f"{pre}:{name}"] = r
            print(pre, name, r, flush=True)
    json.dump(out, open(f"results/scatter_lj_drift_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
