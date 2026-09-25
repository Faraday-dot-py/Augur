"""Kernel-agnostic test of the dual-tree error estimator: labels come from whatever pair force is plugged in (scripts/kernels.py).
Kernels: analytic softened gravity (baseline / regression check), the learned CentralForceDynamics force
(checkpoints/gravity_central_v1.pt), 2D inverse-distance, and a screened (Yukawa, length 30) force. For each kernel the estimator is trained on
its own exact node-pair errors (train states uniform/t0/t1000) and evaluated on held-out flyby states, together with the estimator
trained on the analytic kernel only ("transfer") and the geometric dual tree; kernel evals per particle at matched error.

Usage: PYTHONPATH=. python scripts/dual_kernels_run.py --out results/dual_kernels.json
"""
import argparse
import json

import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import kernels


def build_kernel(name, dev, ckpt):
    return {"analytic": kernels.analytic, "inv_distance": kernels.inv_distance, "yukawa": kernels.yukawa}[name]() if name != "learned" else kernels.learned(ckpt, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--kernels", default="analytic,learned,inv_distance,yukawa")
    ap.add_argument("--ckpt", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--train-states", default="uniform,flyby_t0,flyby_t1000")
    ap.add_argument("--test-states", default="flyby_t5000,flyby_t10000")
    ap.add_argument("--extra-eval", default="uniform")
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--lmax", type=int, default=18)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--theta-max", type=float, default=1.2)
    ap.add_argument("--label-thetas", type=float, nargs="+", default=[0.35, 0.7, 1.2])
    ap.add_argument("--n-pairs", type=int, default=40000)
    ap.add_argument("--n-big", type=int, default=4000)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.9])
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--geo-thetas", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.7])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-targets", type=int, default=10000)
    ap.add_argument("--kernel-feats", type=int, default=0)
    ap.add_argument("--ctrl-iters", type=int, default=0)
    ap.add_argument("--ctrl-tol", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--ctrl-mode", default="pair")
    ap.add_argument("--ctrl-rel", type=float, default=0.01)
    ap.add_argument("--audit-k", type=int, default=1000)
    ap.add_argument("--out", default="results/dual_kernels.json")
    args = ap.parse_args()
    de.KERNEL_FEATS = bool(args.kernel_feats)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4738)
    tr_s, te_s, ex_s = args.train_states.split(","), args.test_states.split(","), args.extra_eval.split(",")
    states, pairs = {}, {}
    for s in tr_s + te_s:
        pos = ao.load_state(s, args, dev)
        tr = ao.Tree(pos, args.lmax)
        fl = dt.Flat(tr)
        states[s] = (pos, tr, fl)
        pairs[s] = de.collect_pairs(tr, fl, args.label_thetas, args.cap, args.n_pairs, args.n_big, gen)
    out = {"args": vars(args), "kernels": {}}
    est_ana = None
    for kname in args.kernels.split(","):
        K = build_kernel(kname, dev, args.ckpt)
        kernels.current = K
        data = {}
        for s, (pos, tr, fl) in states.items():
            ga, gb = pairs[s]
            y, _, _ = de.label(tr, fl, ga, gb, gen)
            x = de.pair_feats(fl, ga, gb)
            keep = torch.isfinite(x).all(1) & torch.isfinite(y)
            data[s] = (x[keep].cpu(), y[keep].cpu())
            print(kname, s, "labels", int(keep.sum()), "y range", float(y.min()), float(y.max()), flush=True)
        xtr, ytr = torch.cat([data[s][0] for s in tr_s]), torch.cat([data[s][1] for s in tr_s])
        xva, yva = torch.cat([data[s][0] for s in te_s]), torch.cat([data[s][1] for s in te_s])
        est = de.train(xtr, ytr, xva, yva, args.taus, dev, args.epochs, 4738)
        if est_ana is None:
            est_ana = est
        heads = {"own": de.Head(est, 0)}
        if est is not est_ana:
            heads["transfer"] = de.Head(est_ana, 0)
        rk = {"fit": {}, "eval": {}}
        for hn, h in heads.items():
            for s in te_s:
                x, y = data[s]
                with torch.no_grad():
                    p = h(x.to(dev)).cpu()
                ss = ((y - y.mean()) ** 2).sum()
                rk["fit"][f"{hn}:{s}"] = {"r2": float(1 - ((y - p) ** 2).sum() / ss), "bias": float((p - y).mean()), "mae": float((y - p).abs().mean())}
                if hn == "own":
                    with torch.no_grad():
                        q = de.Head(est, 1)(x.to(dev)).cpu()
                    rk["fit"][f"{hn}:{s}"]["cover_q90"] = float((y <= q).double().mean())
        print(kname, "fit", json.dumps(rk["fit"]), flush=True)
        for s in te_s + ex_s:
            pos, tr, fl = states[s] if s in states else (None, None, None)
            if pos is None:
                continue
            N = pos.shape[0]
            S = torch.randperm(N, generator=torch.Generator().manual_seed(4738))[:args.eval_targets].to(dev)
            a_ex = K.exact_accel(pos, S, 128)
            r = {"geo": [], "est": []}
            for th in args.geo_thetas:
                a_s, n_m2l, n_dir = dt.dual_accel(tr, fl, th, args.cap)
                a = torch.empty_like(a_s)
                a[tr.order] = a_s
                row = {"theta": th, "cost": (n_m2l + n_dir) / N, **de.metrics2(a[S], a_ex), "net_ratio": ao.net_ratio(a)}
                r["geo"].append(row)
                print(kname, s, "geo", json.dumps(row), flush=True)
            a_c, _, _ = dt.dual_accel(tr, fl, args.theta_max, args.cap)
            fs_node = de.node_scale(fl, a_c)
            for hn, h in heads.items():
                for tol in args.tols:
                    row, _ = de.run_est(tr, fl, S, a_ex, args.cap, h, 1.0, tol, fs_node, args.theta_max)
                    row.update({"head": hn, "tol": tol})
                    r["est"].append(row)
                    print(kname, s, "est", json.dumps(row), flush=True)
            if args.ctrl_iters:
                r["ctrl"] = {}
                for hn, h in heads.items():
                    lam, rows = 1.0, []
                    for it in range(args.ctrl_iters):
                        row, col = de.run_est(tr, fl, S, a_ex, args.cap, h, lam, args.ctrl_tol, fs_node, args.theta_max)
                        if args.ctrl_mode == "e2e":
                            viol = de.audit_e2e(pos, de.run_est.last_a, args.audit_k, gen)
                            over = viol > args.ctrl_rel
                        else:
                            viol = de.audit(tr, fl, col, fs_node, args.ctrl_tol, 3000, gen)
                            over = viol > args.delta
                        row.update({"lam": lam, "violation": viol, "iter": it})
                        rows.append(row)
                        lam = lam * 1.5 if over else lam / 1.1
                    r["ctrl"][hn] = rows
                    last = rows[-5:]
                    print(kname, s, "ctrl", hn, "lam", [round(x["lam"], 2) for x in rows], "viol", [round(x["violation"], 2) for x in rows],
                          "last5 rel_l2", sum(x["rel_l2"] for x in last) / 5, "abs_p99", sum(x["abs_p99"] for x in last) / 5, "cost", sum(x["cost"] for x in last) / 5, flush=True)
            rk["eval"][s] = r
        out["kernels"][kname] = rk
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
