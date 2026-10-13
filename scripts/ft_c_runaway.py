"""Runaway metrics for option-C checkpoints on cached eval sets: per-scene err@20/@100 (mean and median), fraction of balls off the grid (|x|,|y| > extent/2)
at tick 100, fraction of scenes with any ball beyond 100 units, tunnel@20/100 from scatter_contact_scenes.metrics.
Usage: PYTHONPATH=. python scripts/ft_c_runaway.py --ckpts a.pt,b.pt --fam c_id --out results/ft_de/runaway.json"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_contact_scenes as cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--fam", default="c_id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    for path in args.ckpts.split(","):
        model = cs.build("C", dev, path)
        e20, e100, off, far, tun20, tun100 = [], [], [], [], [], []
        for seed in cs.EVAL_SEEDS:
            es = cs.evalset(args.fam, seed, 100, dev)
            Pm, Vm, mask = cs.roll(model, es["scenes"], 100, dev)
            Pt = es["Pt"]
            d = np.linalg.norm(Pm - Pt, axis=-1)
            cnt = mask.sum(1)
            e = (d * mask[:, None]).sum(2) / cnt[:, None]
            e20 += list(e[:, 20])
            e100 += list(e[:, 100])
            half = model.extent / 2
            offb = (np.abs(Pm[:, 100]) > half).any(-1) & (mask > 0)
            off += list(offb.sum(1) / cnt)
            far += list((np.abs(Pm).max(axis=(1, 2, 3)) > 100).astype(float))
            tm, _, _ = cs.metrics(es["scenes"], Pm, Vm, mask)
            tun20.append(float(tm[20]))
            tun100.append(float(tm[100]))
        r = dict(err20_mean=float(np.mean(e20)), err20_median=float(np.median(e20)), err100_mean=float(np.mean(e100)), err100_median=float(np.median(e100)),
                 offgrid_frac_t100=float(np.mean(off)), scenes_any_ball_gt100=float(np.mean(far)), tunnel20=float(np.mean(tun20)), tunnel100=float(np.mean(tun100)), n_scenes=len(e20))
        res[path] = r
        print(path, json.dumps(r), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
