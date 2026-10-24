"""Hole-9 three-arm comparison (docs/theory/heterogeneous-tiered.md): A dense wide U-Net, B tiled + ghosts + far path, C tiers.
Matched domain 128 box, Exp E weights where applicable, seeds 4738 then held-out 9100/9200/9300.

  python scripts/tiered_arms.py run --arm A --model wide:<E.pt> --out results/tiered/arms_A.json
  python scripts/tiered_arms.py run --arm B --model tiled:<E.pt>,nt=4,tile=32,halo=16,far=1 --out results/tiered/arms_B.json
  python scripts/tiered_arms.py run --arm C --model tiered:<nets.pt> --out results/tiered/arms_C.json
  python scripts/tiered_arms.py compare --dir results/tiered

Metrics per arm and seed: far-field force error vs the exact direct sum (all particles; far-dominated subset where the part of the exact
force from sources beyond R0 is > half the total; error of the model explaining that far part), near-field regression (isolated 40-unit
scenes, force error vs exact), 100-step rollout err@k and dE/dp drift on the multi-clump scene, and cost/memory vs N and vs empty fraction.
Pre-registered verdict (doc): PASS if C beats B on far-field accuracy (paired gain > 2 SE over held-out seeds) at matched cost (ms_C <= 1.25 ms_B
at N=2000) without near-field regression (C_near - B_near <= 2 SE), or if C matches B accuracy (|gain| <= 2 SE) while gaining sparsity
(C at 12.5% occupancy <= 0.7x C at 100% and <= B at 12.5%). KILL: neither axis. Otherwise MIXED."""
import argparse
import json
import math
import os

import numpy as np
import torch

from scripts import tiered_eval as te
from scripts import tiered_iface as ti
from scripts import tiered_scenes as sc
from scripts import tiered_truth as tt

R0 = 20.0
SPAN = 128.0
FAR_ICS = (("multiclump", 300), ("clumpy", 400))
NEAR_ICS = (("plummer", 200), ("cold_collapse", 200))
HELD = (9100, 9200, 9300)


def settled_force(model, b, warm=3):
    st = model.init_state(b["pos"].shape[0], b["pos"].device)
    ctx = dict(phys=b["phys"])
    for _ in range(warm):
        a, st = model.force(b["pos"], b["vel"], b["mass"], b["mask"], st, ctx)
    return a


def far_metrics(model, seed, dev):
    out = {}
    num = den = nums = dens = fnum = fden = 0.0
    for ic, n in FAR_ICS:
        b = te.grav_batch(ic, n, SPAN, seed, 4, dev)
        a_t = tt.grav_acc(b["pos"], b["mass"], b["mask"])
        a_near = ti.RangeLower(R0)(b["pos"], b["vel"], b["mass"], b["mask"])
        a_far = a_t - a_near
        a_m = settled_force(model, b)
        err = ((a_m - a_t) ** 2).sum(-1)
        dom = ((a_far ** 2).sum(-1) > 0.25 * (a_t ** 2).sum(-1)).to(err.dtype) * b["mask"]
        num, den = num + float(err.sum()), den + float((a_t ** 2).sum())
        nums, dens = nums + float((err * dom).sum()), dens + float(((a_t ** 2).sum(-1) * dom).sum())
        fnum, fden = fnum + float(err.sum()), fden + float((a_far ** 2).sum())
        out[ic] = dict(rel_rms=float(math.sqrt(err.sum() / (a_t ** 2).sum())), far_dominated_fraction=float(dom.sum() / b["mask"].sum()))
    out["rel_rms_all"] = math.sqrt(num / den)
    out["rel_rms_far_dominated"] = math.sqrt(nums / max(dens, 1e-30))
    out["err_over_far_magnitude"] = math.sqrt(fnum / fden)
    return out


def near_metrics(model, seed, dev):
    out, num, den = {}, 0.0, 0.0
    for ic, n in NEAR_ICS:
        b = te.grav_batch(ic, n, 40.0, seed, 4, dev)
        a_t = tt.grav_acc(b["pos"], b["mass"], b["mask"])
        a_m = settled_force(model, b)
        e = float(((a_m - a_t) ** 2).sum())
        out[ic] = math.sqrt(e / float((a_t ** 2).sum()))
        num, den = num + e, den + float((a_t ** 2).sum())
    out["rel_rms"] = math.sqrt(num / den)
    return out


def cost_metrics(model, seed, dev, quick):
    rng = np.random.default_rng(seed)
    out = {"n_sweep": {}, "empty_sweep": {}}
    mk = lambda n, w: dict(pos=torch.as_tensor(rng.uniform(-w, w, (1, n, 2)), device=dev),
                           vel=torch.as_tensor(rng.normal(0, 0.3, (1, n, 2)), device=dev), mass=torch.full((1, n), 0.05, device=dev, dtype=torch.float64),
                           mask=torch.ones(1, n, device=dev, dtype=torch.float64), phys=tt.Physics(grav=dict(eps=tt.EPS)))
    for n in ((500, 2000) if quick else (500, 1000, 2000, 4000)):
        t, mem = te.timed_step(model, mk(n, SPAN / 2 * 0.95))
        out["n_sweep"][str(n)] = dict(ms=t, peak_mb=mem)
    for f in ((1.0, 0.125) if quick else (1.0, 0.25, 0.125)):
        t, mem = te.timed_step(model, mk(2000, SPAN / 2 * 0.95 * math.sqrt(f)))
        out["empty_sweep"][str(f)] = dict(ms=t, peak_mb=mem)
    return out


def run(args, dev):
    model = te.load_model(args.model, dev, 0.1)
    res = dict(arm=args.arm, model=model.name, spec=args.model, seeds=args.seeds, r0=R0, span=SPAN, per_seed={})
    for s in args.seeds:
        r = dict(far=far_metrics(model, s, dev), near=near_metrics(model, s, dev))
        b = te.grav_batch("multiclump", 300, SPAN, s, 4, dev)
        r["rollout"] = te.compare(model, b, 20 if args.quick else 100, ks=(10, 20, 50, 100))
        r["cost"] = cost_metrics(model, s, dev, args.quick)
        res["per_seed"][str(s)] = r
        print(args.arm, s, "far", round(r["far"]["rel_rms_far_dominated"], 4), "near", round(r["near"]["rel_rms"], 4), flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(te.tolist(res), open(args.out, "w"), indent=1)


def paired(a, b):
    """mean and standard error of a - b over seeds."""
    d = np.array(a) - np.array(b)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")


def arm_vals(d, f):
    return [f(d["per_seed"][str(s)]) for s in HELD if str(s) in d["per_seed"]]


def compare(args):
    arms = {}
    for a in "ABC":
        p = os.path.join(args.dir, f"arms_{a}.json")
        if os.path.exists(p):
            arms[a] = json.load(open(p))
    rep = dict(arms={a: v["model"] for a, v in arms.items()})
    far = lambda r: r["far"]["rel_rms_far_dominated"]
    near = lambda r: r["near"]["rel_rms"]
    ms = lambda key: (lambda r: r["cost"]["n_sweep"][key]["ms"])
    ms_e = lambda f: (lambda r: r["cost"]["empty_sweep"][f]["ms"])
    for a, d in arms.items():
        rep[a] = dict(far_dominated_rel_rms=float(np.mean(arm_vals(d, far))), all_rel_rms=float(np.mean(arm_vals(d, lambda r: r["far"]["rel_rms_all"]))),
                      near_rel_rms=float(np.mean(arm_vals(d, near))), err_at_100=float(np.mean(arm_vals(d, lambda r: r["rollout"]["err"]["100"]))),
                      ms_N2000=float(np.mean(arm_vals(d, ms("2000")))), ms_occ_1=float(np.mean(arm_vals(d, ms_e("1.0")))),
                      ms_occ_0125=float(np.mean(arm_vals(d, ms_e("0.125")))))
    if "B" not in arms:
        rep["verdict"] = "pending: arm B missing"
    elif "C" not in arms:
        rep["verdict"] = "pending: arm C (tiers) not run; A and B are baselines"
    else:
        B, C = arms["B"], arms["C"]
        gain_m, gain_se = paired([x for x in arm_vals(B, far)], arm_vals(C, far))
        rel_gain = gain_m / float(np.mean(arm_vals(B, far)))
        near_m, near_se = paired(arm_vals(C, near), arm_vals(B, near))
        cost_ratio = rep["C"]["ms_N2000"] / rep["B"]["ms_N2000"]
        sparse_ratio = rep["C"]["ms_occ_0125"] / rep["C"]["ms_occ_1"]
        acc_better = gain_m > 2 * gain_se
        acc_equal = abs(gain_m) <= 2 * gain_se
        no_regress = near_m <= 2 * near_se
        axis1 = acc_better and cost_ratio <= 1.25 and no_regress
        axis2 = acc_equal and sparse_ratio <= 0.7 and rep["C"]["ms_occ_0125"] <= rep["B"]["ms_occ_0125"] and no_regress
        rep["criteria"] = dict(far_gain_rel=rel_gain, far_gain_abs=gain_m, far_gain_se=gain_se, near_C_minus_B=near_m, near_se=near_se,
                               cost_ratio_C_over_B_N2000=cost_ratio, C_sparse_ratio_0125_over_1=sparse_ratio, axis1_accuracy_at_matched_cost=bool(axis1),
                               axis2_matched_accuracy_with_sparsity=bool(axis2), near_regression=not no_regress)
        rep["verdict"] = "PASS" if (axis1 or axis2) else ("KILL" if not acc_better and not (sparse_ratio <= 0.7) else "MIXED")
    print(json.dumps(te.tolist(rep), indent=1))
    json.dump(te.tolist(rep), open(os.path.join(args.dir, "arms_compare.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("--arm", choices=list("ABC"))
    ap.add_argument("--model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[4738, *HELD])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/tiered/arms.json")
    ap.add_argument("--dir", default="results/tiered")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.mode == "run":
        run(args, torch.device(args.device))
    else:
        compare(args)


if __name__ == "__main__":
    main()
