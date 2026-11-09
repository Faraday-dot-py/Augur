"""Held-out reconstruction error by cell for a saved mesh SAE (curiosity, no pass/kill). Reports normalized MSE overall, with the largest-error cells
removed, and where the largest-error cells come from (law, scene, radius, input norm).
Usage: PYTHONPATH=. python scripts/sae_outlier_cells.py --ckpt ckpt/main_s4738.pt --layer dec3 --sae results/sae_interp_lowdead_dec3_channelnorm_s4738_sae.pt --key mesh_64 --out results/sae_outlier_cells_dec3_channelnorm_s4738.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_multi as sm
from scripts.sae_interp import LAWS, encode, mesh_acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--heldout-scenes", type=int, default=6)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = sm.build().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    S = torch.load(a.sae, map_location="cpu")[a.key]
    sae = (S["W"].to(dev), S["We"].to(dev), S["be"].to(dev), S["bd"].to(dev), None, S["method"], S["k"])
    mu = S["mu"].to(dev)
    sc = S["sc"].to(dev) if torch.is_tensor(S["sc"]) else S["sc"]
    X, lab, cont, sid = mesh_acts(model, dev, a.scenes, 400, 3, a.seed, a.layer)
    scene = sid % a.scenes
    te = scene > scene.max() - a.heldout_scenes
    Xn = ((X.to(dev) - mu) / sc)[te.to(dev)]
    rec = encode(sae, Xn) @ sae[0] + sae[3]
    err = ((rec - Xn) ** 2).sum(1).cpu()
    den = float(((Xn - Xn.mean(0)) ** 2).sum())
    order = torch.argsort(err, descending=True)
    n = len(err)
    res = {"ckpt": a.ckpt, "layer": a.layer, "key": a.key, "n_heldout": n, "nmse": float(err.sum()) / den, "total_error_share_top": {}, "nmse_without_top": {},
           "err_percentiles": {str(p): float(np.percentile(err.numpy(), p)) for p in (50, 90, 99, 99.9)}}
    for frac in (0.001, 0.01, 0.05):
        m = max(1, int(n * frac))
        res["total_error_share_top"][str(frac)] = float(err[order[:m]].sum() / err.sum())
        keep = torch.ones(n, dtype=torch.bool)
        keep[order[:m]] = False
        xk = Xn[keep.to(dev)]
        res["nmse_without_top"][str(frac)] = float(err[keep].sum()) / float(((xk - xk.mean(0)) ** 2).sum())
    lt, rad, nrm = lab[te], cont["radius_from_centre"][te], Xn.norm(dim=-1).cpu()
    top = order[:max(1, int(n * 0.01))]
    res["top1pct_law_counts"] = {LAWS[i]: int((lt[top] == i).sum()) for i in range(len(LAWS))}
    res["top1pct_radius_median"], res["all_radius_median"] = float(rad[top].median()), float(rad.median())
    res["top1pct_input_norm_median"], res["all_input_norm_median"] = float(nrm[top].median()), float(nrm.median())
    res["top1pct_scene_counts"] = {int(s): int((scene[te][top] == s).sum()) for s in scene[te].unique()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
