"""genker static test: per kernel train an own estimator (est_train.train_estimator, labels from that kernel's own coarse-vs-fine gap),
save it, and compare with the analytic-trained estimator (transfer) and the geometric dual tree on held-out flyby t5000/t10000 and uniform:
fit (R2, bias, q90 coverage) and kernel evals per particle at matched rel_l2 0.01 / 0.02.

Usage: PYTHONPATH=. python scripts/genker_static.py --kernels lj,softgrav2 --out results/genker_static_a.json
"""
import argparse
import json
import math

import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import est_train
from scripts import genker_kernels as gk
from scripts import kernels


def matched(rows, target):
    pts = sorted((r["rel_l2"], r["cost"]) for r in rows)
    for (e0, c0), (e1, c1) in zip(pts, pts[1:]):
        if e0 <= target <= e1 and e1 > e0:
            w = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp(math.log(c0) + w * (math.log(c1) - math.log(c0)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="lj,softgrav2,pow15,inv_distance,yukawa30,learned")
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--train-states", default="uniform,flyby_t0,flyby_t1000")
    ap.add_argument("--test-states", default="flyby_t5000,flyby_t10000,uniform")
    ap.add_argument("--fit-states", default="flyby_t5000,flyby_t10000")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--lmax", type=int, default=18)
    ap.add_argument("--theta-max", type=float, default=1.2)
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--geo-thetas", type=float, nargs="+", default=[0.2, 0.25, 0.35, 0.5, 0.7, 1.0])
    ap.add_argument("--eval-targets", type=int, default=10000)
    ap.add_argument("--analytic-est", default="checkpoints/est_analytic.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4738)
    states = {}
    for s in sorted(set(args.test_states.split(",") + args.fit_states.split(","))):
        pos = ao.load_state(s, args, dev)
        tr = ao.Tree(pos, args.lmax)
        fl = dt.Flat(tr)
        ga, gb = de.collect_pairs(tr, fl, [0.35, 0.7, 1.2], args.cap, 40000, 4000, gen)
        states[s] = (pos, tr, fl, ga, gb)
    transfer = est_train.load(args.analytic_est, dev)
    out = {"args": vars(args), "kernels": {}}
    for kname in args.kernels.split(","):
        K = gk.make_kernel(kname, dev)
        kernels.current = K
        own_est = est_train.train_estimator(K, dev)
        est_train.save(own_est, f"checkpoints/genker_est_{kname}.pt", kname)
        own = de.Head(own_est, 0)
        q90 = de.Head(own_est, 1)
        rk = {"fit": {}, "eval": {}}
        for s in args.fit_states.split(","):
            pos, tr, fl, ga, gb = states[s]
            y, _, _ = de.label(tr, fl, ga, gb, gen)
            x = de.pair_feats(fl, ga, gb)
            keep = torch.isfinite(x).all(1) & torch.isfinite(y)
            x, y = x[keep], y[keep]
            ss = ((y - y.mean()) ** 2).sum()
            with torch.no_grad():
                for hn, h in (("own", own), ("transfer", transfer)):
                    p = h(x)
                    d = {"r2": float(1 - ((y - p) ** 2).sum() / ss), "bias": float((p - y).mean()), "n": int(keep.sum()), "nonfinite_labels": int((~keep).sum())}
                    if hn == "own":
                        d["cover_q90"] = float((y <= q90(x)).double().mean())
                    else:
                        d["cover_q90"] = float((y <= transfer.est.net(((x - transfer.est.mu) / transfer.est.sd).float())[:, 1].double()).double().mean())
                    rk["fit"][f"{hn}:{s}"] = d
        print(kname, "fit", json.dumps(rk["fit"]), flush=True)
        for s in args.test_states.split(","):
            pos, tr, fl, _, _ = states[s]
            N = pos.shape[0]
            S = torch.randperm(N, generator=torch.Generator().manual_seed(4738))[:args.eval_targets].to(dev)
            a_ex = K.exact_accel(pos, S, 128)
            r = {"geo": [], "own": [], "transfer": []}
            for th in args.geo_thetas:
                a_s, n_m2l, n_dir = dt.dual_accel(tr, fl, th, args.cap)
                a = torch.empty_like(a_s)
                a[tr.order] = a_s
                r["geo"].append({"theta": th, "cost": (n_m2l + n_dir) / N, **de.metrics2(a[S], a_ex), "net_ratio": ao.net_ratio(a)})
            a_c, _, _ = dt.dual_accel(tr, fl, args.theta_max, args.cap)
            fs = de.node_scale(fl, a_c)
            for hn, h in (("own", own), ("transfer", transfer)):
                for tol in args.tols:
                    row, _ = de.run_est(tr, fl, S, a_ex, args.cap, h, 1.0, tol, fs, args.theta_max)
                    row["tol"] = tol
                    r[hn].append(row)
            r["matched"] = {str(t): {m: matched(r[m], t) for m in ("geo", "own", "transfer")} for t in (0.01, 0.02)}
            print(kname, s, "matched", json.dumps(r["matched"]), flush=True)
            rk["eval"][s] = r
        out["kernels"][kname] = rk
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
