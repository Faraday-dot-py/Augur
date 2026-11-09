"""Locates the bodies where the default and adaptdeep pair-term outputs differ on the collapsed Plummer cloud and brute-forces them in fp64 against all N bodies.
Cloud construction is identical to scripts/pair_term_adaptive_depth_test.py (sample('plummer', n, default_rng(seed), core_scale), unit masses, float32).
Memory: pos 12 MB x N/1e6 x 2 (fp32 + fp64 copy), two outputs 24 MB each, difference ranking a few 12 MB vectors; the default call spills to the all-bodies fallback (about 11 queries x N x 2 fp32 per chunk, as in job 7235);
the fp64 brute force holds N x (2 + 1 + 1) doubles for one body at a time, about 150 MB of fp64 at N = 3e6.
Usage: PYTHONPATH=. python scripts/pair_term_adaptdeep_bruteforce_at_3e6.py --ckpt kern/E_frozen.pt --device cuda --out-dir results/pair_term_adaptdeep_bruteforce_at_3e6
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.pair_term_adaptive_depth_test import make_model
from scripts.pm_baseline_force_error import sample


def topk_nearest(p, q, k):
    d = p[None, :, :] - p[q][:, None]
    r_all = torch.sqrt((d ** 2).sum(-1) + 1e-8)
    bad = torch.arange(p.shape[0], device=p.device)[None] == q[:, None]
    r_sel, i = torch.topk(r_all + 1e6 * bad.to(p.dtype), k, dim=-1, largest=False)
    return d, r_sel, i


def force_from_set(model64, p64, m64, qi, idx):
    d = p64[idx] - p64[qi][:, None]
    r = torch.sqrt((d ** 2).sum(-1) + 1e-8)
    return model64._pp_force(d, r, m64[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--core-scale", type=float, default=0.25)
    ap.add_argument("--ckpt", default="random")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--topk-bodies", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pos = torch.tensor(sample("plummer", a.n, rng, a.core_scale), dtype=torch.float32, device=dev)[None]
    n = pos.shape[1]
    assert n == a.n, (n, a.n)
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    model = make_model(sf, a.ckpt, dev)
    knn = model.knn
    res = {"n": n, "core_scale": a.core_scale, "ckpt": a.ckpt, "device": a.device, "seed": a.seed, "knn": knn}
    with torch.no_grad():
        t0 = time.time()
        model.opt = {"adapt"}
        out_default = model.pp_acc(pos, mass, mask, knn)
        res["default_call_s"] = time.time() - t0
        model.opt = {"adapt", "adaptdeep"}
        t0 = time.time()
        out_deep = model.pp_acc(pos, mass, mask, knn)
        res["adaptdeep_call_s"] = time.time() - t0
        print("pair-term calls done", res["default_call_s"], res["adaptdeep_call_s"], flush=True)
        diff = (out_default - out_deep)[0]
        dn = diff.norm(dim=-1)
        res["max_abs_elementwise_diff"] = float(diff.abs().max())
        res["max_norm_diff"] = float(dn.max())
        res["count_diff_gt_1e-6"] = int((dn > 1e-6).sum())
        res["count_diff_gt_1e-4"] = int((dn > 1e-4).sum())
        res["count_diff_gt_1e-2"] = int((dn > 1e-2).sum())
        rms = float(out_deep[0].norm(dim=-1).pow(2).mean().sqrt())
        res["adaptdeep_rms_acc"] = rms
        k = min(a.topk_bodies, n)
        top_v, top_i = torch.topk(dn, k)
        res["argmax_body"] = int(top_i[0])
        model64 = copy.deepcopy(model).double()
        p32, p64 = pos[0], pos[0].double()
        m64 = mass[0].double()
        bodies = []
        for j in range(k):
            b = int(top_i[j])
            q = torch.tensor([b], device=dev)
            d64, r64, i64 = topk_nearest(p64, q, knn + 1)
            _, r32, i32 = topk_nearest(p32, q, knn + 1)
            f64 = force_from_set(model64, p64, m64, q, i64[:, :knn])[0]
            f32set = force_from_set(model64, p64, m64, q, i32[:, :knn])[0]
            dflt, deep = out_default[0, b].double(), out_deep[0, b].double()
            set64 = set(i64[0, :knn].tolist())
            set32 = set(i32[0, :knn].tolist())
            bodies.append({
                "body": b, "abs_diff_default_vs_adaptdeep": float(top_v[j]),
                "err_default_vs_fp64_bruteforce": float((dflt - f64).norm()),
                "err_adaptdeep_vs_fp64_bruteforce": float((deep - f64).norm()),
                "err_default_vs_fp32_set_force": float((dflt - f32set).norm()),
                "err_adaptdeep_vs_fp32_set_force": float((deep - f32set).norm()),
                "fp64_top16_set_differs_from_fp32_top16_set": set64 != set32,
                "fp32_set_force_vs_fp64_set_force": float((f32set - f64).norm()),
                "r16_fp64": float(r64[0, knn - 1]), "r17_fp64": float(r64[0, knn]), "gap_r17_minus_r16_fp64": float(r64[0, knn] - r64[0, knn - 1]),
                "r16_fp32": float(r32[0, knn - 1]), "r17_fp32": float(r32[0, knn]), "gap_r17_minus_r16_fp32": float(r32[0, knn] - r32[0, knn - 1]),
                "ref_force_norm": float(f64.norm()),
            })
            print(bodies[-1], flush=True)
        res["bodies"] = bodies
        res["argmax"] = bodies[0]
    res["done"] = True
    with open(os.path.join(a.out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out_dir, "DONE"), "w") as f:
        f.write("DONE\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
