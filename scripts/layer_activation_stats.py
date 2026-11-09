"""Per-layer activation statistics of the mesh U-Net at mass-bearing cells (curiosity): fraction of exactly-zero outputs, mean |activation| and
per-channel dead count, for every enc and dec block. Usage: PYTHONPATH=. python scripts/layer_activation_stats.py --ckpt ckpt/main_s4738.pt --out results/layer_activation_stats_s4738.json
"""
import argparse
import json

import torch

from scripts import scatter_multi as sm
from scripts.sae_interp import mesh_acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = sm.build().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    out = {"ckpt": a.ckpt}
    for kind, n in (("enc", 6), ("dec", 5)):
        for i in range(n):
            X, _, _, _ = mesh_acts(model, dev, 4, 400, 3, a.seed, f"{kind}{i}")
            r = {"mean_abs": float(X.abs().mean()), "frac_exact_zero": float((X == 0).float().mean()), "dead_channels": int((X.abs().amax(0) == 0).sum()), "channels": X.shape[1],
                 "max_abs": float(X.abs().max())}
            out[f"{kind}{i}"] = r
            print(kind, i, r, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
