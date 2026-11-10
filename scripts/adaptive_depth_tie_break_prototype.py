"""CPU smoke test of a lower-index tie-break within relative tol at the k-th neighbour of pp_acc_adapt, without editing scripts/scatter_field.py.
Builds a patched copy of scatter_field in memory (the three torch.topk neighbour selections in pp_acc_adapt call tie_topk instead) and compares it to the unpatched module:
(1) synthetic rows with planted ties, candidate columns shuffled: unpatched pick depends on column order, patched pick is always the lowest global index;
(2) Plummer cloud: patched output must be bit-identical to unpatched on every body whose rank-k and rank-(k+1) fp32 distances are not within tol; cost ratio of the call;
(3) square lattice (massive exact ties at rank 16): default vs adaptdeep vs a dense reference applying the same rule.
Usage: PYTHONPATH=. python scripts/adaptive_depth_tie_break_prototype.py --out results/adaptive_depth_tie_and_single_query_diff_investigation/tie_break_prototype.json
"""
import argparse
import json
import os
import time
import types

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.pair_term_adaptive_depth_test import make_model
from scripts.pm_baseline_force_error import sample

TIE_TOPK_SRC = '''
TIE_STATS = {"rows": 0, "straddle": 0, "overflow": 0}


def tie_topk(r, gid, kk, tol=1e-6, extra=4):
    ke = min(kk + extra, r.shape[1])
    rv, i = torch.topk(r, ke, dim=-1, largest=False)
    if ke == kk:
        return rv, i
    rk = rv[:, kk - 1:kk]
    grp = (rv - rk).abs() <= tol * rk
    straddle = grp[:, kk:].any(1) & (rk[:, 0] < 1e5)
    TIE_STATS["rows"] += rv.shape[0]
    TIE_STATS["straddle"] += int(straddle.sum())
    TIE_STATS["overflow"] += int((straddle & grp[:, -1] & (ke < r.shape[1])).sum())
    if not straddle.any():
        return rv[:, :kk], i[:, :kk]
    g = torch.gather(gid, 1, i)
    pos = torch.arange(ke, device=r.device)[None].expand_as(g)
    cls = torch.where(grp, 1, torch.where(rv < rk, 0, 2))
    comb = cls.long() * (1 << 40) + torch.where(grp, g, pos)
    sel = torch.argsort(comb, dim=1)[:, :kk]
    sel = torch.gather(sel, 1, torch.argsort(torch.gather(rv, 1, sel), dim=1, stable=True))
    rv2, i2 = torch.gather(rv, 1, sel), torch.gather(i, 1, sel)
    return torch.where(straddle[:, None], rv2, rv[:, :kk]), torch.where(straddle[:, None], i2, i[:, :kk])
'''


def patched_module():
    src = open(sf.__file__).read()
    a = "r_sel, i = torch.topk(r_all + 1e6 * bad.to(pos.dtype), kk, dim=-1, largest=False)"
    b = "r_sel, i = torch.topk(r_all + 1e6 * bad.to(pos.dtype), k, dim=-1, largest=False)\n                ds = torch.gather(d, 1, i[..., None].expand(-1, -1, 2))\n                qs.append(qq)"
    assert src.count(a) == 1 and src.count(b) == 1, (src.count(a), src.count(b))
    src = src.replace(a, "r_sel, i = tie_topk(r_all + 1e6 * bad.to(pos.dtype), cand, kk)")
    src = src.replace(b, "r_sel, i = tie_topk(r_all + 1e6 * bad.to(pos.dtype), torch.arange(n, device=dev)[None].expand(len(qq), -1), k)\n                ds = torch.gather(d, 1, i[..., None].expand(-1, -1, 2))\n                qs.append(qq)")
    mod = types.ModuleType("scatter_field_tie")
    mod.__file__ = sf.__file__
    exec(compile(TIE_TOPK_SRC + "\n" + src, "scatter_field_tie", "exec"), mod.__dict__)
    return mod


def synthetic(mod):
    g = torch.Generator().manual_seed(4738)
    rows, w, kk = 2000, 40, 16
    r = torch.rand(rows, w, generator=g) + 4.0
    r[:, :15] = torch.rand(rows, 15, generator=g) + 1.0
    r[:, 15] = r[:, 16] = 3.0
    r = torch.gather(r, 1, torch.stack([torch.randperm(w, generator=g) for _ in range(rows)]))
    gid = torch.stack([torch.randperm(w, generator=g) for _ in range(rows)])
    base_rv, base_i = torch.topk(r, kk, dim=-1, largest=False)
    pr_rv, pr_i = mod.tie_topk(r, gid, kk)
    tie_cols = (r == 3.0).nonzero()[:, 1].reshape(rows, 2)
    want = torch.gather(gid, 1, tie_cols).min(1).values
    in_set = (torch.gather(gid, 1, pr_i) == want[:, None]).any(1)
    plain_in = (torch.gather(gid, 1, base_i) == want[:, None]).any(1)
    return {"rows": rows, "patched_picks_lowest_index_frac": float(in_set.float().mean()), "unpatched_picks_lowest_index_frac": float(plain_in.float().mean())}


def plummer(mod, n, core, dev):
    rng = np.random.default_rng(4738)
    pos = torch.tensor(sample("plummer", n, rng, core), dtype=torch.float32, device=dev)[None]
    mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
    out = {}
    m0, m1 = make_model(sf, "random", dev), make_model(mod, "random", dev)
    with torch.no_grad():
        for name, opt in (("default", {"adapt"}), ("adaptdeep", {"adapt", "adaptdeep"})):
            m0.opt, m1.opt = set(opt), set(opt)
            t = time.time()
            a = m0.pp_acc(pos, mass, mask, 16)
            ta = time.time() - t
            mod.TIE_STATS.update({"rows": 0, "straddle": 0, "overflow": 0})
            t = time.time()
            b = m1.pp_acc(pos, mass, mask, 16)
            tb = time.time() - t
            neq = (a != b).any(-1)[0]
            p = pos[0].double()
            tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(p.numpy())
            dd, ii = tree.query(p.numpy(), k=19)
            r = np.sqrt(dd ** 2 + 1e-8)
            tie_bodies = torch.tensor(np.abs(r[:, 17] - r[:, 16]) <= 1e-6 * r[:, 16])
            out[name] = {"n": n, "unpatched_s": ta, "patched_s": tb, "ratio": tb / ta, "bodies_with_changed_output": int(neq.sum()), "changed_bodies": [{"i": int(j), "fp64_rel_gaps_r15_16_r16_17_r17_18": [float((r[j, k] - r[j, k - 1]) / r[j, k - 1]) for k in (16, 17, 18)]} for j in neq.nonzero()[:, 0].tolist()], "bodies_with_fp64_rank16_17_within_tol": int(tie_bodies.sum()),
                         "changed_bodies_not_tie_bodies": int((neq & ~tie_bodies).sum()), "tie_stats": dict(mod.TIE_STATS)}
    return out


def lattice(mod, side, dev):
    s = 0.01
    gx, gy = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    g = torch.Generator().manual_seed(4738)
    pos = (torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).float() * s)
    pos = pos[torch.randperm(pos.shape[0], generator=g)][None].to(dev)
    n = pos.shape[1]
    mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
    m0, m1 = make_model(sf, "random", dev), make_model(mod, "random", dev)
    res = {"n": n}
    with torch.no_grad():
        outs = {}
        for label, m in (("unpatched", m0), ("patched", m1)):
            for name, opt in (("default", {"adapt"}), ("adaptdeep", {"adapt", "adaptdeep"})):
                m.opt = set(opt)
                outs[(label, name)] = m.pp_acc(pos, mass, mask, 16)
        scale = float(outs[("patched", "default")].norm(dim=-1).max())
        for k1, k2 in ((("unpatched", "default"), ("unpatched", "adaptdeep")), (("patched", "default"), ("patched", "adaptdeep")), (("unpatched", "default"), ("patched", "default"))):
            d = (outs[k1] - outs[k2]).norm(dim=-1)[0]
            res[f"{k1}_vs_{k2}"] = {"max_abs_diff": float(d.max()), "bodies_diff_gt_1e-3_of_max_force": int((d > 1e-3 * scale).sum()), "bodies_bitdiff": int((outs[k1] != outs[k2]).any(-1).sum())}
        res["force_scale_max"] = scale
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--core-scale", type=float, default=0.25)
    ap.add_argument("--lattice-side", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cpu")
    mod = patched_module()
    res = {"synthetic": synthetic(mod)}
    print(res, flush=True)
    res["plummer"] = plummer(mod, a.n, a.core_scale, dev)
    print(res["plummer"], flush=True)
    res["lattice"] = lattice(mod, a.lattice_side, dev)
    print(res["lattice"], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
