"""Evaluate the trained staggered-double-grid-with-quadratic-deposit candidate against its control (frozen the frozen single-law gravity checkpoint) on identical scenes.
Per seed (9100/9200/9300 or --seeds): expB 8 scenes + cluster 4 scenes, free rollout 100 ticks vs fp64 truth; err@k, dE@k (|E_model - E_truth| / |E_truth| at the same tick), shift probe
(probe_half: median relative force change under half-cell x, y, diagonal rigid shifts; probe_off: quarter-cell diagonal and (0.37, 0.11) cells) and seconds per tick, for both models in the same process.
Usage: PYTHONPATH=. python scripts/staggered_double_grid_eval.py --control E_ms_kp_pot_v_g128.pt --cand ckpt.pt --variant ms_kp_pot_v_g128_quad_dual --out out.json [--seeds 12000,12100,12200] [--smoke]"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_error_growth as seg
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import shift_average_stage1 as s1
from scripts import train_scatter_field as tsf


def build_candidate(variant, ckpt, dev):
    kw = {**tsf.BASE, **tsf.VARIANTS[variant]}
    model = sf.ScatterField(extent=tsf.EXPS["single-law-gravity-checkpoint-g128"]["extent"], dt=seg.DT, in_scale=tsf.EXPS["single-law-gravity-checkpoint-g128"]["in_scale"], **kw)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
    model.load_state_dict(sd)
    return model.to(dev).eval()


def run_model(model, scenes, dev):
    rows, cost, ph, po = [], [], [], []
    for sc in scenes:
        net = s1.ShiftAvg(model, s1.PASSES[1], dev)
        ta = time.time()
        Pm, Vm = seg.roll(net, sc, steps=s1.TR)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        cost.append((time.time() - ta) / s1.TR)
        e, de = s1.curves(sc, Pm, Vm)
        rows.append({**{f"err@{t}": float(e[t]) for t in s1.KS}, **{f"dE@{t}": float(de[t]) for t in s1.KS}, "finite": float(np.isfinite(e).all() and np.isfinite(de).all())})
        ph.append(s1.probe(model, sc, s1.PROBE_HALF, dev)[1])
        po.append(s1.probe(model, sc, s1.PROBE_OFF, dev)[1])
    res = {n: float(np.mean([r[n] for r in rows])) for n in rows[0]}
    res.update(probe_half=float(np.mean(ph)), probe_off=float(np.mean(po)), sec_per_tick=float(np.mean(cost)), n_scenes=len(rows))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--variant", default="ms_kp_pot_v_g128_quad_dual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="9100,9200,9300")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cpu" if args.smoke else "cuda")
    if args.smoke:
        s1.TR, s1.KS, seg.T = 10, (5, 10, 10), 10
        torch.set_num_threads(2)
    seg.DEV = dev
    torch.manual_seed(4738)
    control = ss.build_model("E", seg.DT, dev, args.control)
    cand = build_candidate(args.variant, args.cand, dev)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"control": args.control, "cand": args.cand, "variant": args.variant, "smoke": args.smoke, "seeds": {}}
    with torch.no_grad():
        for seed in seeds[:1] if args.smoke else seeds:
            scenes = []
            for fam, n in s1.FAMS.items():
                scenes += seg.make_scenes(fam, 1 if args.smoke else n, seed)
            out["seeds"][str(seed)] = {"control": run_model(control, scenes, dev), "cand": run_model(cand, scenes, dev)}
            print(f"seed {seed} {len(scenes)} scenes", flush=True)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
