"""Delete-the-star probe on a trained scatter-field checkpoint, dense in distance, saved for rendering.

Usage: PYTHONPATH=. python scripts/delete_star_trace.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --tag delete_star_E
"""
import argparse

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", default="delete_star")
ap.add_argument("--mstar", type=float, default=20.0)
ap.add_argument("--warm", type=int, default=15)
ap.add_argument("--after", type=int, default=25)
args = ap.parse_args()
dev = torch.device("cuda")
model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
model.eval()
dists = tuple(float(d) for d in np.linspace(1.0, 28.0, 28))
res = sf.delete_star_probe(model, dev, dists=dists, warm=args.warm, after=args.after, mstar=args.mstar)
print("lag steps", res["lag_steps"])
print("ratio_before", [round(x, 2) for x in res["ratio_before"]])
np.savez(f"results/{args.tag}.npz", dists=np.array(res["dists"]), warm=args.warm, mstar=args.mstar, truth_before=np.array(res["a_truth_before"]),
         delete=np.array(res["delete"]), control=np.array(res["control"]))
