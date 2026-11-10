"""GPU check of the lowest-index tie-break at 3e6 bodies (collapsed Plummer cloud, seed 4738, core scale 0.25, kern/E_frozen.pt), unpatched vs patched pp_acc_adapt, default and adaptdeep paths.
The patch is an in-memory copy of scripts/scatter_field.py built by scripts/adaptive_depth_tie_break_prototype.py (patched_module); scripts/scatter_field.py itself is not edited.
Reference for every number: the unpatched output of the same path on the same scene and seed. Ambiguous set: bodies whose fp64 rank-16/17 relative gap is <= 1e-6, read from the CPU kd-tree list in
results/adaptive_depth_tie_and_single_query_diff_investigation/tie_count_3e6.json (counting rule: gap = (r17 - r16) / r16, r = sqrt(d^2 + 1e-8), fp64 on the fp32 coordinates; a body counts when gap <= 1e-6; lowest body index wins inside a tied group).
Verdict by hand from result.json keys verdict_input.*; see the pre-registration in docs/debugging/experiment-log.md (adaptive-depth-tie-break-gpu-check-3e6).
Usage: PYTHONPATH=. python scripts/adaptive_depth_tie_break_gpu_check_3e6.py --ckpt kern/E_frozen.pt --device cuda --ambiguous results/adaptive_depth_tie_and_single_query_diff_investigation/tie_count_3e6.json --out-dir results/adaptive_depth_tie_break_gpu_check_3e6
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.adaptive_depth_3e6_argmax_body_brute_force import captured_call, indices_from_ds
from scripts.adaptive_depth_tie_break_prototype import patched_module
from scripts.pair_term_adaptive_depth_test import make_model
from scripts.pm_baseline_force_error import sample

REL_TOL = 1e-6
TIME_RATIO_LINE = 1.5
PATHS = (("default", {"adapt"}), ("adaptdeep", {"adapt", "adaptdeep"}))


def timed_call(model, pos, mass, mask, knn, dev, reps):
    ts = []
    out = None
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        out = model.pp_acc(pos, mass, mask, knn)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t)
    return out, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--core-scale", type=float, default=0.25)
    ap.add_argument("--ckpt", default="random")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--body", type=int, default=1381234)
    ap.add_argument("--ambiguous", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--default-reps", type=int, default=1)
    ap.add_argument("--deep-reps", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)
    amb = json.load(open(a.ambiguous))
    ambiguous = sorted(amb["ambiguous_bodies_gap_le_1e-6"])
    band = sorted(amb["boundary_band_bodies_gap_gt_1e-6_le_1.5e-6"])
    if a.n == 3000000:
        assert len(ambiguous) == 105 and a.body in ambiguous, (len(ambiguous), a.body)
    else:
        ambiguous = [i for i in ambiguous if i < a.n]
        band = [i for i in band if i < a.n]
    rng = np.random.default_rng(a.seed)
    pos = torch.tensor(sample("plummer", a.n, rng, a.core_scale), dtype=torch.float32, device=dev)[None]
    n = pos.shape[1]
    assert n == a.n, (n, a.n)
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    mod = patched_module()
    base = make_model(sf, a.ckpt, dev)
    patched = make_model(mod, a.ckpt, dev)
    knn = base.knn
    amb_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    amb_mask[torch.tensor(ambiguous, device=dev)] = True
    band_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    if band:
        band_mask[torch.tensor(band, device=dev)] = True
    res = {"n": n, "core_scale": a.core_scale, "ckpt": a.ckpt, "device": a.device, "seed": a.seed, "knn": knn, "rel_tol": REL_TOL, "time_ratio_line": TIME_RATIO_LINE,
           "ambiguous_count": len(ambiguous), "boundary_band_count": len(band), "paths": {}}
    outs = {}
    with torch.no_grad():
        for m in (base, patched):
            for o in ({"adapt", "adaptdeep"}, {"adapt"}):
                m.opt = set(o)
                m.pp_acc(pos[:, :200000], mass[:, :200000], mask[:, :200000], knn)
        for name, opt in PATHS:
            reps = a.default_reps if name == "default" else a.deep_reps
            base.opt = set(opt)
            patched.opt = set(opt)
            if reps > 1 or name != "default":
                timed_call(base, pos, mass, mask, knn, dev, 1)
                timed_call(patched, pos, mass, mask, knn, dev, 1)
            ob, tb = timed_call(base, pos, mass, mask, knn, dev, reps)
            mod.TIE_STATS.update({"rows": 0, "straddle": 0, "overflow": 0})
            op, tp = timed_call(patched, pos, mass, mask, knn, dev, reps)
            ts = {k: int(v) // reps for k, v in mod.TIE_STATS.items()}
            outs[name] = (ob, op)
            d = (op - ob)[0].norm(dim=-1)
            scale = ob[0].norm(dim=-1)
            rel = d / scale.clamp(min=1e-30)
            bit = (op != ob).any(-1)[0]
            outside = ~amb_mask
            res["paths"][name] = {
                "unpatched_call_s": tb, "patched_call_s": tp, "time_ratio": float(np.median(tp) / np.median(tb)),
                "tie_stats_per_call": ts,
                "bodies_bit_changed": int(bit.sum()), "bodies_bit_changed_ambiguous": int((bit & amb_mask).sum()), "bodies_bit_changed_outside_ambiguous": int((bit & outside).sum()),
                "max_rel_change_outside_ambiguous": float(rel[outside].max()), "max_abs_change_outside_ambiguous": float(d[outside].max()),
                "bodies_rel_change_gt_tol_outside_ambiguous": int((rel[outside] > REL_TOL).sum()),
                "indices_rel_change_gt_tol_outside_ambiguous": [int(i) for i in (rel > REL_TOL).logical_and(outside).nonzero()[:, 0]][:50],
                "bodies_rel_change_gt_tol_outside_ambiguous_and_band": int((rel[outside & ~band_mask] > REL_TOL).sum()),
                "bodies_rel_change_gt_tol_in_boundary_band": int((rel[band_mask] > REL_TOL).sum()) if band else 0,
                "max_rel_change_ambiguous": float(rel[amb_mask].max()),
            }
            print(name, {k: v for k, v in res["paths"][name].items() if k != "unpatched_call_s"}, flush=True)
        dp, dd = outs["default"][1], outs["adaptdeep"][1]
        bit_pp = (dp != dd).any(-1)[0]
        dif = (dp - dd)[0].norm(dim=-1)
        res["patched_default_vs_patched_adaptdeep"] = {"bodies_bit_different": int(bit_pp.sum()), "max_abs_diff": float(dif.max()), "indices_bit_different": [int(i) for i in bit_pp.nonzero()[:, 0]][:50],
                                                  "max_rel_diff": float((dif / dp[0].norm(dim=-1).clamp(min=1e-30)).max()), "bodies_rel_diff_gt_tol": int((dif / dp[0].norm(dim=-1).clamp(min=1e-30) > REL_TOL).sum())}
        res["unpatched_default_vs_unpatched_adaptdeep"] = {"bodies_bit_different": int((outs["default"][0] != outs["adaptdeep"][0]).any(-1).sum())}

        b = a.body
        qmask = torch.zeros(1, n, device=dev)
        qmask[0, b] = 1
        p32 = pos[0]
        pick = {}
        for label, model in (("unpatched", base), ("patched", patched)):
            for name, opt in PATHS:
                _, (ds, rs), stats = captured_call(model, set(opt), pos, mass, mask, qmask, knn)
                idx = indices_from_ds(p32, b, ds)
                pick[f"{label}_{name}"] = {"top16": idx, "selected_16th_index": idx[knn - 1], "in_top16": sorted(idx)}
        tied_group = [627953, 849166] if a.n == 3000000 else None
        lower = min(tied_group) if tied_group else None
        res["body_1381234_pick"] = pick
        res["body_1381234_check"] = {
            "tied_group": tied_group, "lowest_index_member": lower,
            "patched_default_equals_patched_adaptdeep_top16": sorted(pick["patched_default"]["top16"]) == sorted(pick["patched_adaptdeep"]["top16"]),
            "patched_16th_both_paths": [pick["patched_default"]["selected_16th_index"], pick["patched_adaptdeep"]["selected_16th_index"]],
            "patched_both_pick_lowest_index_member": lower in pick["patched_default"]["top16"] and lower in pick["patched_adaptdeep"]["top16"] and tied_group[1] not in pick["patched_default"]["top16"] and tied_group[1] not in pick["patched_adaptdeep"]["top16"] if tied_group else False,
        }
        print(res["body_1381234_check"], flush=True)
        pd, pa = res["paths"]["default"], res["paths"]["adaptdeep"]
        res["verdict_input"] = {
            "patched_default_bit_identical_to_patched_adaptdeep": res["patched_default_vs_patched_adaptdeep"]["bodies_bit_different"] == 0,
            "non_ambiguous_bodies_over_tol_default": pd["bodies_rel_change_gt_tol_outside_ambiguous"],
            "non_ambiguous_bodies_over_tol_adaptdeep": pa["bodies_rel_change_gt_tol_outside_ambiguous"],
            "time_ratio_default": pd["time_ratio"], "time_ratio_adaptdeep": pa["time_ratio"],
            "time_ratio_over_line": pd["time_ratio"] > TIME_RATIO_LINE or pa["time_ratio"] > TIME_RATIO_LINE,
            "body_pick_same_in_both_paths": res["body_1381234_check"]["patched_default_equals_patched_adaptdeep_top16"],
            "body_16th_is_lowest_index_member_both_paths": res["body_1381234_check"]["patched_both_pick_lowest_index_member"],
            "patched_default_vs_adaptdeep_bodies_rel_diff_gt_tol": res["patched_default_vs_patched_adaptdeep"]["bodies_rel_diff_gt_tol"],
            "ambiguous_set_ceiling_1.5e-6_variant_over_tol": [pd["bodies_rel_change_gt_tol_outside_ambiguous_and_band"], pa["bodies_rel_change_gt_tol_outside_ambiguous_and_band"]],
        }
    res["done"] = True
    with open(os.path.join(a.out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out_dir, "DONE"), "w") as f:
        f.write("DONE\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
