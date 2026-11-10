"""CPU smoke: is the single-query qmask force different from the full-call force because of batch shape (matmul and sum order inside _pp_force), independent of which neighbours were picked?
Captures (ds, r_sel, mj) per _pp_force call in a full pp_acc call, then (1) re-evaluates _pp_force on the same rows with different row counts and compares bits, (2) compares neighbour
index sets and forces of qmask single-body calls against the full call, (3) reports the same for float64 (shape effects should vanish to ~1e-16).
Usage: PYTHONPATH=. python scripts/adaptive_depth_single_query_diff_cpu.py --out results/adaptive_depth_tie_and_single_query_diff_investigation/single_query_diff_cpu.json
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts.pair_term_adaptive_depth_test import make_model
from scripts.pm_baseline_force_error import sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--core-scale", type=float, default=0.25)
    ap.add_argument("--bodies", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cpu")
    rng = np.random.default_rng(4738)
    pos = torch.tensor(sample("plummer", a.n, rng, a.core_scale), dtype=torch.float32)[None]
    mass, mask = torch.ones(1, a.n), torch.ones(1, a.n)
    model = make_model(sf, "random", dev)
    model.opt = {"adapt", "adaptdeep"}
    res = {"n": a.n, "core_scale": a.core_scale}
    bodies = np.sort(rng.choice(a.n, a.bodies, replace=False))
    with torch.no_grad():
        cap = []
        orig = model._pp_force

        def hook(ds, r_sel, mj):
            out = orig(ds, r_sel, mj)
            cap.append((ds.clone(), r_sel.clone(), mj.clone(), out.clone()))
            return out

        model._pp_force = hook
        full = model.pp_acc(pos, mass, mask, 16)
        del model._pp_force
        ds, r_sel, mj, out = max(cap, key=lambda c: c[0].shape[0])
        rows = ds.shape[0]
        res["largest_captured_call_rows"] = rows
        res["captured_calls"] = len(cap)
        shape_res = {}
        for m in (1, 2, 7, 64, 1000, rows):
            m = min(m, rows)
            sub = torch.cat([orig(ds[c:c + m], r_sel[c:c + m], mj[c:c + m]) for c in range(0, min(rows, 4000), m)])
            ref = out[:sub.shape[0]]
            shape_res[str(m)] = {"rows_compared": sub.shape[0], "rows_bit_different": int((sub != ref).any(-1).sum()), "max_abs_diff": float((sub - ref).abs().max())}
        res["pp_force_rechunked_vs_original_call_fp32"] = shape_res
        mlp_in = model._feat(r_sel[:2000], "p")
        w_all = model.ppmlp(mlp_in)[..., 0]
        w_one = torch.cat([model.ppmlp(mlp_in[i:i + 1])[..., 0] for i in range(2000)])
        res["ppmlp_one_row_vs_batch_bit_different_rows"] = int((w_all != w_one).any(-1).sum())
        term = (w_all[..., None] * ds[:2000] / r_sel[:2000, :, None])
        s_all = term.sum(1)
        s_one = torch.cat([term[i:i + 1].sum(1) for i in range(2000)])
        res["sum_over_16_one_row_vs_batch_bit_different_rows"] = int((s_all != s_one).any(-1).sum())
        diffs, set_diffs, bitdiff = [], 0, 0
        for b in bodies:
            q = torch.zeros(1, a.n)
            q[0, b] = 1
            cap.clear()
            model._pp_force = hook
            out1 = model.pp_acc(pos, mass, mask, 16, qmask=q)
            del model._pp_force
            d = (out1[0, b] - full[0, b]).norm()
            diffs.append(float(d))
            bitdiff += int(not torch.equal(out1[0, b], full[0, b]))
        res["qmask_vs_full_fp32"] = {"bodies": len(bodies), "bit_different": bitdiff, "max_abs_diff": max(diffs), "median_abs_diff": float(np.median(diffs)), "force_norm_median": float(full[0, bodies].norm(dim=-1).median())}
        m64 = model.double()
        p64, ma64, mk64 = pos.double(), mass.double(), mask.double()
        full64 = m64.pp_acc(p64, ma64, mk64, 16)
        bd64, mx64 = 0, 0.0
        for b in bodies[:50]:
            q = torch.zeros(1, a.n, dtype=torch.float64)
            q[0, b] = 1
            o = m64.pp_acc(p64, ma64, mk64, 16, qmask=q)
            bd64 += int(not torch.equal(o[0, b], full64[0, b]))
            mx64 = max(mx64, float((o[0, b] - full64[0, b]).norm()))
        res["qmask_vs_full_fp64"] = {"bodies": 50, "bit_different": bd64, "max_abs_diff": mx64}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
