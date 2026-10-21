"""Edge population of the Exp E training distribution (hypothesis: edge cells never populated in training).
Regenerates the Exp E train set (cfg E: expB + cluster mix, seed 4738, 50 ticks) at a reduced scene count and reports the fraction of body-ticks with
max-norm coordinate beyond several radii, the per-scene maximum, and the fraction of scenes ever touching the last 5 cells (|coord| > 29.5)."""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    cfg = tsf.EXPS["E"]
    data = tsf.make_data(cfg, args.scenes, args.steps, 4738, dev, 0.1, 0.5)
    P, V, M, mask = sf.to_tensors(data, dev)
    c = P.abs().amax(-1)
    w = mask[:, None].expand_as(c)
    res = {"scenes": args.scenes, "body_ticks": float(w.sum())}
    for r in (20.0, 26.0, 28.0, 29.5, 31.0, 32.0):
        res[f"frac_body_ticks_coord_gt_{r}"] = float(((c > r).double() * w).sum() / w.sum())
    res["frac_scenes_touch_gt_29.5"] = float((((c > 29.5).double() * w).amax((1, 2)) > 0).double().mean())
    res["frac_scenes_touch_gt_26"] = float((((c > 26.0).double() * w).amax((1, 2)) > 0).double().mean())
    res["frac_scenes_any_outside_32"] = float((((c > 32.0).double() * w).amax((1, 2)) > 0).double().mean())
    res["n_bodies_mean"] = float(mask.sum(1).mean())
    res["n_bodies_max"] = float(mask.sum(1).max())
    print("POP " + json.dumps(res), flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
