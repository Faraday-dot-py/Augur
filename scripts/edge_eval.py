"""Evaluate edge-fix checkpoints: held-out err@5/10/20 (expB, cluster, open-domain in-box) on seeds 9000/12000 plus the edge probes (scripts/edge_probe.py)
under several inference configs.

Usage: python scripts/edge_eval.py --models E=ckpt.pt,open_full=ckpt2.pt --variants E=ms_kp_pot_v_g128,... --configs base=default,far05=default+far@0.05 --out X.json
Config opts use '+' instead of ',' (e.g. default+far@0.05+kext). A '-' prefix on a model name's variant is not needed: variant is chosen per model with --variants
(default ms_kp_pot_v_g128; ghost arms use ms_kp_pot_v_g128_ghost).
"""
import argparse
import json

import numpy as np
import torch

from scripts import edge_probe as ep
from scripts import gravity_sim as gs
from scripts import scatter_edge_data as sed
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

SEEDS = (9000, 12000)
TICKS = (5, 10, 20)
DT, EPS = 0.1, 0.5


def err_curve(model, data, dev, box_mask=False):
    P, V, M, mask = sf.to_tensors(data, dev)
    Pm, _ = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), P.shape[1] - 1)
    e = (Pm.double() - P).norm(dim=-1)
    w = mask[:, None].expand_as(e)
    if box_mask:
        w = w * (P.abs().amax(-1) < model.extent / 2).double()
    return {str(k): float((e * w).sum((0, 2))[k] / w.sum((0, 2))[k]) for k in TICKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--variants", default="")
    ap.add_argument("--configs", default="base=default")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    vmap = dict(x.split("=") for x in args.variants.split(",") if x)
    cfgs = [x.split("=") for x in args.configs.split(",")]
    sets = {}
    for sd in SEEDS:
        sets[("expB", sd)] = gs.make_dataset(48, (10, 100), 100, sd, scale=True, device=dev, dt=DT, eps=EPS)
        sets[("cluster", sd)] = sf.make_cluster_dataset(48, 100, sd + 1, dev, dt=DT, eps=EPS)
        sets[("open", sd)] = sed.make_open_dataset(48, 50, sd + 2, dev)
    ec, tc, res = {}, {}, {}
    for spec in args.models.split(","):
        name, ck = spec.split("=")
        res[name] = {}
        for cname, copts in cfgs:
            opts = copts.replace("+", ",")
            model = tsf.build(tsf.EXPS["E"], vmap.get(name, "ms_kp_pot_v_g128"), DT).to(dev)
            sf.load_compat(model, torch.load(ck, map_location=dev, weights_only=False)["model"])
            model.eval()
            sf.apply_opts(model, opts)
            r = {"err": {}}
            for (kind, sd), data in sets.items():
                r["err"][f"{kind}_{sd}"] = err_curve(model, data, dev, box_mask=(kind == "open"))
            line = " ".join(f"{k}:{v['5']:.4f}/{v['10']:.4f}/{v['20']:.4f}" for k, v in r["err"].items())
            print(f"EVAL {name} [{cname}] err@5/10/20 {line}", flush=True)
            if not args.no_probe:
                pm = tsf.build(tsf.EXPS["E"], vmap.get(name, "ms_kp_pot_v_g128"), DT).to(dev)
                sf.load_compat(pm, torch.load(ck, map_location=dev, weights_only=False)["model"])
                pm.eval()
                sf.apply_opts(pm, opts + ",-graph")
                ops.chunked_pp(pm)
                r["static"] = [ep.static_probe(pm, dev, 100000, s, ec) for s in SEEDS]
                r["rollout"] = [ep.rollout_probe(pm, dev, 20000, 2000, 100, s, tc) for s in SEEDS]
                for st in r["static"]:
                    print(f"PROBE {name} [{cname}] static seed {st['seed']} ratio {st['band_ratio']:.1f} signmatch {st['sign_match']:.2f} changes {st['sign_changes_per_dir']:.1f} "
                          f"interior_err {st['interior_rel_err']:.3f}", flush=True)
                for ro in r["rollout"]:
                    print(f"PROBE {name} [{cname}] roll seed {ro['seed']} beyond70/400 {ro['model']['beyond70_per400']:.1f} rev {ro['model']['reversal_frac']:.4f} "
                          f"(truth {ro['truth']['beyond70_per400']:.1f} / {ro['truth']['reversal_frac']:.4f}) median_r {ro['model']['median_r']:.1f} p99_r {ro['model']['p99_r']:.1f} truth {ro['truth']['median_r']:.1f}/{ro['truth']['p99_r']:.1f}", flush=True)
            res[name][cname] = r
            json.dump(res, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
