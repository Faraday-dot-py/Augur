"""Dumps the neighbour sets of one body (default: the job 7297 argmax body 1381234) on the collapsed 3e6 Plummer cloud for three paths: default adapt pair term, adaptdeep, and fp64 brute force.
Scene, model and calls are identical to scripts/pair_term_adaptdeep_bruteforce_at_3e6.py. The full default and adaptdeep calls are rerun to confirm the argmax body and to get the reference forces;
the neighbour indices of each path come from a second call with qmask set to that one body (the cell list is built from all bodies, so the candidate lists are the same), whose force must equal the full-call force.
Tie rule for the verdict: a body in the symmetric difference of two sets is a tie member when |r - r16_fp64| / r16_fp64 <= 1e-6. Reference tie-break: lower body index wins on equal fp64 distance.
Usage: PYTHONPATH=. python scripts/adaptive_depth_3e6_argmax_body_brute_force.py --ckpt kern/E_frozen.pt --device cuda --out-dir results/adaptive_depth_3e6_argmax_body_brute_force
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
from scripts.pair_term_adaptdeep_bruteforce_at_3e6 import force_from_set

TIE_REL = 1e-6
RANKS = (14, 15, 16, 17, 18)


def captured_call(model, opt, pos, mass, mask, qmask, knn):
    cap = []
    orig = model._pp_force

    def hook(ds, r_sel, mj):
        cap.append((ds.clone(), r_sel.clone()))
        return orig(ds, r_sel, mj)

    model._pp_force = hook
    model.opt = opt
    try:
        out = model.pp_acc(pos, mass, mask, model.knn, qmask=qmask)
    finally:
        del model._pp_force
    assert len(cap) == 1 and cap[0][0].shape[0] == 1, [c[0].shape for c in cap]
    return out, cap[0], dict(model.adapt_stats)


def indices_from_ds(p32, b, ds):
    d_all = p32 - p32[b]
    idx = []
    for row in ds[0]:
        hit = (d_all == row).all(-1).nonzero()[:, 0]
        assert hit.numel() == 1, hit.tolist()
        idx.append(int(hit[0]))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000000)
    ap.add_argument("--core-scale", type=float, default=0.25)
    ap.add_argument("--ckpt", default="random")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--body", type=int, default=1381234)
    ap.add_argument("--expect-argmax", type=int, default=1)
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
    b = a.body
    res = {"n": n, "core_scale": a.core_scale, "ckpt": a.ckpt, "device": a.device, "seed": a.seed, "knn": knn, "body": b, "tie_rel": TIE_REL}
    with torch.no_grad():
        model.opt = {"adapt"}
        t0 = time.time()
        out_default = model.pp_acc(pos, mass, mask, knn)
        res["default_call_s"] = time.time() - t0
        model.opt = {"adapt", "adaptdeep"}
        t0 = time.time()
        out_deep = model.pp_acc(pos, mass, mask, knn)
        res["adaptdeep_call_s"] = time.time() - t0
        dn = (out_default - out_deep)[0].norm(dim=-1)
        res["argmax_body_this_run"] = int(dn.argmax())
        res["abs_diff_default_vs_adaptdeep_at_body"] = float(dn[b])
        res["argmax_matches_body"] = res["argmax_body_this_run"] == b
        if a.expect_argmax:
            assert res["argmax_matches_body"], (res["argmax_body_this_run"], b)
        print("full calls done", res["default_call_s"], res["adaptdeep_call_s"], flush=True)

        qmask = torch.zeros(1, n, device=dev)
        qmask[0, b] = 1
        p32, p64, m64 = pos[0], pos[0].double(), mass[0].double()
        model64 = copy.deepcopy(model).double()
        q = torch.tensor([b], device=dev)

        d64 = p64 - p64[b]
        r64_all = torch.sqrt((d64 ** 2).sum(-1) + 1e-8)
        r64_all[b] = 1e6
        order = torch.argsort(r64_all)
        top64 = order[:knn + 4]
        r64 = r64_all[top64]
        r16 = float(r64[knn - 1])
        set64 = [int(i) for i in top64[:knn]]
        group = ((r64_all - r16).abs() / r16 <= TIE_REL).nonzero()[:, 0]
        group = group[group != b]
        res["fp64"] = {
            "ranks": {str(k): {"index": int(top64[k - 1]), "r": float(r64[k - 1])} for k in RANKS},
            "gap_r17_minus_r16": float(r64[knn] - r64[knn - 1]),
            "gap_rel_r17_minus_r16": float((r64[knn] - r64[knn - 1]) / r64[knn - 1]),
            "top16": set64,
            "tie_group_indices_within_tie_rel_of_r16": sorted(int(i) for i in group),
            "tie_group_r": {str(int(i)): float(r64_all[i]) for i in group},
            "tie_break_rule_used": "torch.argsort of fp64 r, no index key (strict distance order; lower index only matters at exactly equal fp64 r)",
            "exactly_equal_r16_r17": bool(r64[knn] == r64[knn - 1]),
        }
        f64 = force_from_set(model64, p64, m64, q, torch.tensor([set64], device=dev))[0]

        paths = {}
        for name, opt, full in (("default", {"adapt"}, out_default), ("adaptdeep", {"adapt", "adaptdeep"}, out_deep)):
            out1, (ds, rs), stats = captured_call(model, opt, pos, mass, mask, qmask, knn)
            idx = indices_from_ds(p32, b, ds)
            r32_all = torch.sqrt(((p32 - p32[b]) ** 2).sum(-1) + 1e-8)
            sel = set(idx)
            sym = sorted(sel.symmetric_difference(set(set64)))
            diff_members = []
            for i in sym:
                rel = abs(float(r64_all[i]) - r16) / r16
                diff_members.append({"index": i, "in_path_set": i in sel, "in_fp64_set": i in set(set64), "r_fp64": float(r64_all[i]), "r_fp32": float(r32_all[i]), "rel_gap_to_r16_fp64": rel, "within_tie": rel <= TIE_REL})
            fb = full[0, b].double()
            paths[name] = {
                "top16": idx,
                "ranks_r_fp32_as_selected": {str(k): float(rs[0, k - 1]) for k in RANKS if k <= knn},
                "selected_16th_index": idx[knn - 1],
                "symmetric_difference_vs_fp64_set": diff_members,
                "sets_equal_to_fp64": sel == set(set64),
                "sets_equal_up_to_ties": all(m["within_tie"] for m in diff_members),
                "worst_rel_gap_in_difference": max([m["rel_gap_to_r16_fp64"] for m in diff_members], default=0.0),
                "qmask_call_force_equals_full_call_force": bool(torch.equal(out1[0, b], full[0, b])),
                "qmask_vs_full_force_abs_diff": float((out1[0, b].double() - fb).norm()),
                "full_call_err_vs_fp64_bruteforce_force": float((fb - f64).norm()),
                "adapt_stats_levels": stats["levels"],
                "adapt_stats_fallback_queries": stats["fallback"],
            }
            print(name, paths[name], flush=True)
        res["paths"] = paths

        r32_all = torch.sqrt(((p32 - p32[b]) ** 2).sum(-1) + 1e-8)
        r32_all[b] = 1e6
        o32 = torch.argsort(r32_all)
        res["fp32_distance_ties_at_16th"] = {
            "r_fp32_ranks": {str(k): {"index": int(o32[k - 1]), "r": float(r32_all[o32[k - 1]])} for k in RANKS},
            "fp32_r16_equals_r17": bool(r32_all[o32[knn - 1]] == r32_all[o32[knn]]),
            "count_bodies_with_fp32_r_equal_to_fp32_r16": int((r32_all == r32_all[o32[knn - 1]]).sum()),
            "indices_with_fp32_r_equal_to_fp32_r16": [int(i) for i in (r32_all == r32_all[o32[knn - 1]]).nonzero()[:, 0]],
        }
        tie_members = res["fp64"]["tie_group_indices_within_tie_rel_of_r16"]
        lower = min(tie_members) if tie_members else None
        res["tie_break_reference"] = {
            "rule": "lower body index wins on equal fp64 distance (applies within the tie group; fp64 distances in the group are not exactly equal, so the fp64 sort picks the strictly nearer member)",
            "lower_index_member_of_tie_group": lower,
            "fp64_sort_picks": res["fp64"]["ranks"]["16"]["index"],
            "default_picks_16th": paths["default"]["selected_16th_index"],
            "adaptdeep_picks_16th": paths["adaptdeep"]["selected_16th_index"],
            "adaptdeep_pick_is_lower_index_member": paths["adaptdeep"]["selected_16th_index"] == lower,
            "default_pick_is_lower_index_member": paths["default"]["selected_16th_index"] == lower,
            "adaptdeep_pick_equals_fp64_pick": paths["adaptdeep"]["selected_16th_index"] == res["fp64"]["ranks"]["16"]["index"],
        }
        ad = paths["adaptdeep"]
        res["verdict_input"] = {
            "adaptdeep_sets_equal_up_to_ties": ad["sets_equal_up_to_ties"],
            "adaptdeep_worst_rel_gap_in_difference": ad["worst_rel_gap_in_difference"],
        }
    res["done"] = True
    with open(os.path.join(a.out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out_dir, "DONE"), "w") as f:
        f.write("DONE\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
