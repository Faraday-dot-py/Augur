"""Time-step lever for the scatter field on LJ (docs/debugging/experiment-log.md, hole-1 follow-ups). For each config (nsub = whole-model substeps per
0.025 step, pair_sub = extra pair-only substeps inside each model step) rolls out from truth frame 0 and reports position error vs truth at frames
1, 5, 20 (lj_eval.err_curve), wall seconds per recorded frame (cuda-synchronised, 2 warmup frames), and optionally 160-frame NVE dE/N.
`exact` configs use an oracle-LJ, mesh-off model (integrator floor). Usage:
PYTHONPATH=. python -m scripts.scatter_lj_dt_sweep --ckpt ckpt/P_final.pt --label P --configs 1:1,2:1,4:1,1:2,1:4 --tag dt_P [--exact]"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import lj_eval
from scripts import scatter_lj as sl


def run_cfg(model, z, nf, dev):
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.time()
    fp, _, st, ok = sl.rollout(model, z["pos"][0].astype(np.float64), z["vel"][0].astype(np.float64), z["box"], nf, dev)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    return fp, st, ok, (time.time() - t0) / nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--configs", default="1:1")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--scenes", default="solid,liquid,coexist,melt")
    ap.add_argument("--drift-scenes", default="gas,liquid,solid,coexist")
    ap.add_argument("--drift-prefixes", default="results/lj")
    ap.add_argument("--drift-frames", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = {}
    for cfg in args.configs.split(","):
        n, ps = [int(x) for x in cfg.split(":")]
        if args.exact:
            model = sl.build_model(energy=True, oracle=True, nomesh=True, substeps=n).to(dev)
            model.pair_sub = ps
        else:
            model = sl.load_model(args.ckpt, dev, nsub=n, pair_sub=ps)
        model.eval()
        r = {}
        for name in args.scenes.split(","):
            z = np.load(f"results/lj_{name}.npz")
            run_cfg(model, z, 2, dev)
            fp, st, ok, spf = run_cfg(model, z, 20, dev)
            r[name] = {"finite": bool(ok), "sec_per_frame": spf, **{f"err{k}": float(lj_eval.err_curve(fp[k], z["pos"][k], z["box"]).mean()) for k in (1, 5, 20) if k < len(fp)}}
        if args.drift_frames:
            for pre in args.drift_prefixes.split(","):
                for name in args.drift_scenes.split(","):
                    z = np.load(f"{pre}_{name}.npz")
                    _, _, st, ok = sl.rollout(model, z["pos"][0].astype(np.float64), z["vel"][0].astype(np.float64), z["box"], args.drift_frames, dev)
                    r[f"drift:{pre}:{name}"] = {"finite": bool(ok), "dE_per_N": float(st[-1, 3] - st[0, 3]), "T_end": float(st[-1, 1])}
        out[cfg] = r
        print(args.label, cfg, json.dumps(r), flush=True)
    json.dump(out, open(f"results/scatter_lj_dt_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
