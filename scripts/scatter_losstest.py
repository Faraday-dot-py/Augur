"""Single-change test of the rollout position loss (squared vs per-body Euclid vs Huber) for scatter-field training, variant ms_kp_pot_v_g128 (pp=2, knn=16).
Loop identical to scripts/scatter_prior_pp8.py (Exp E data: 2000 scenes x 30 ticks, seed 4738, batch 16, k 4->20, cosine lr, same iterations).
Setups: scratch (lr 1e-3, random init seed 4738) or warm (lr 5e-4, init = Exp B checkpoint, fresh Adam: E's warm-start recipe).
Eval seeds 9100/9200/9300 (48 scenes x 100 ticks each: first 24 = 10-200-body scale-init 'ab', last 24 clusters 'cl').
Usage: PYTHONPATH=. python scripts/scatter_losstest.py --setup warm --arm euclid --iters 4000 --init-b <B ckpt>"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as ts
from scripts.gap_probe import acc_of, exact_acc, nn_dist

EPS = 0.5
DT = 0.1
CFG = ts.EXPS["E"]
VARIANT = "ms_kp_pot_v_g128"
SEEDS = (9100, 9200, 9300)
NBINS = [("<0.5", 0.0, 0.5), ("0.5-1", 0.5, 1.0), (">=1", 1.0, 1e9)]
DEV = torch.device("cuda")


def train_tensors():
    return sf.to_tensors(ts.make_data(CFG, 2000, 30, 4738, DEV, DT, EPS), DEV)


@torch.no_grad()
def window_errs(model, tensors, k=20, starts=(0, 5, 10), n_scenes=96, stride=21, bs=16):
    P, V, M, mask = [t.float() for t in tensors]
    P, V, M, mask = P[::stride][:n_scenes], V[::stride][:n_scenes], M[::stride][:n_scenes], mask[::stride][:n_scenes]
    errs = []
    for t0 in starts:
        for i in range(0, P.shape[0], bs):
            Pw, Vw, m, mk = P[i:i + bs, t0:t0 + k + 1], V[i:i + bs, t0:t0 + k + 1], M[i:i + bs], mask[i:i + bs]
            pos, vel = Pw[:, 0], Vw[:, 0]
            field = model.init_field(pos.shape[0], DEV)
            for s in range(k):
                pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
                errs.append((pos - Pw[:, s + 1]).norm(dim=-1)[mk > 0].cpu())
    return torch.cat(errs)


@torch.no_grad()
def step_curve(model, tensors, k=20, starts=(0, 5, 10), n_scenes=96, stride=21, bs=16):
    P, V, M, mask = [t.float() for t in tensors]
    P, V, M, mask = P[::stride][:n_scenes], V[::stride][:n_scenes], M[::stride][:n_scenes], mask[::stride][:n_scenes]
    per = [[] for _ in range(k)]
    for t0 in starts:
        for i in range(0, P.shape[0], bs):
            Pw, Vw, m, mk = P[i:i + bs, t0:t0 + k + 1], V[i:i + bs, t0:t0 + k + 1], M[i:i + bs], mask[i:i + bs]
            pos, vel = Pw[:, 0], Vw[:, 0]
            field = model.init_field(pos.shape[0], DEV)
            for s in range(k):
                pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
                per[s].append((pos - Pw[:, s + 1]).norm(dim=-1)[mk > 0].cpu())
    return [float(torch.cat(x).median()) for x in per]


def force_bins(model, P, V, M, mask):
    E, T, K, Vd = [], [], [], []
    for t in (0, 10, 20, 30):
        pos, vel = P[:, t].float(), V[:, t].float()
        a = acc_of(model, pos, vel, M.float(), mask.float())
        at = exact_acc(pos, M.float(), mask.float())
        E.append(((a - at).double() ** 2).sum(-1))
        T.append((at.double() ** 2).sum(-1))
        K.append(nn_dist(P[:, t], mask))
        Vd.append(mask)
    E, T, K, Vd = torch.stack(E), torch.stack(T), torch.stack(K), torch.stack(Vd)
    out = {"all": float(((E * Vd).sum() / (T * Vd).sum()).sqrt())}
    for nm, lo, hi in NBINS:
        sel = (K >= lo) & (K < hi) & (Vd > 0)
        out[nm] = float((E[sel].sum() / T[sel].sum()).sqrt()) if sel.sum() else None
    return out


def evaluate(model):
    res = {}
    for seed in SEEDS:
        data = ts.make_data(CFG, 48, 100, seed, DEV, DT, EPS)
        for tag, sub in (("ab", data[:24]), ("cl", data[24:])):
            m, (Pm, Vm, Pt, Vt) = ts.evaluate_model(model, sub, DEV, DT, EPS)
            P, V, M, mask = sf.to_tensors(sub, DEV)
            e = (Pm[:, 20] - Pt[:, 20]).norm(dim=-1)[mask > 0].double().cpu()
            q = torch.quantile(e, torch.tensor([0.5, 0.9, 0.99], dtype=torch.double))
            top = e.pow(2).sort(descending=True).values
            res[f"{tag}_{seed}"] = {
                "err": [m["err"][i] for i in (4, 9, 19, 49, 99)],
                "e20": {"mean": float(e.mean()), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                        "sq_loss": float((e ** 2).mean() / 2), "top1pct_sq_share": float(top[: max(1, len(top) // 100)].sum() / top.sum())},
                "force": force_bins(model, P, V, M, mask),
                "dE100": m["energy_drift_rel_model"][100], "dE100_true": m["energy_drift_rel_true"][100],
            }
            r = res[f"{tag}_{seed}"]
            print(f"{tag} {seed} err@5/10/20/50/100 {[round(x, 5) for x in r['err']]} e20 med/mean/p90/p99 {r['e20']['median']:.4f}/{r['e20']['mean']:.4f}/{r['e20']['p90']:.4f}/{r['e20']['p99']:.4f} force {r['force']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["scratch", "warm"], required=True)
    ap.add_argument("--arm", choices=["sq", "euclid", "huber", "delta", "vel_euclid", "stepw", "huber_unit"], required=True)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--init-b", default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/losstest")
    ap.add_argument("--ckpt", default="checkpoints/losstest")
    ap.add_argument("--eval-ckpt", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.arm == "delta":
        model = ts.build(CFG, VARIANT, DT).to(DEV)
        model.load_state_dict(torch.load(args.init_b, map_location=DEV, weights_only=False)["model"])
        model.eval()
        e = window_errs(model, train_tensors())
        sc = step_curve(model, train_tensors())
        d = {"step_median": sc, "delta": float(e.median()), "mean": float(e.mean()), "p90": float(e.quantile(0.9)), "p99": float(e.quantile(0.99)), "n": len(e)}
        print("B-ckpt per-body Euclid error over steps 1..20, train scenes:", d, flush=True)
        json.dump(d, open(f"{args.out}/delta.json", "w"))
        return
    if args.arm in ("huber", "huber_unit") and args.delta is None:
        args.delta = json.load(open(f"{args.out}/delta.json"))["delta"]
    key = f"{args.setup}_{args.arm}{args.tag}_{args.iters}"
    out_path = f"{args.out}/{key}.json"
    model = ts.build(CFG, VARIANT, DT).to(DEV)
    r = {"setup": args.setup, "arm": args.arm, "iters": args.iters, "seed": args.seed}
    if args.eval_ckpt:
        model.load_state_dict(torch.load(args.eval_ckpt, map_location=DEV, weights_only=False)["model"])
        out_path = f"{args.out}/{key}_evalonly.json"
    else:
        if args.setup == "warm":
            model.load_state_dict(torch.load(args.init_b, map_location=DEV, weights_only=False)["model"])
        r["delta"] = args.delta
        tensors = train_tensors()
        rl = {"vel_euclid": "euclid", "stepw": "euclid"}.get(args.arm, args.arm)
        step_err = torch.tensor(json.load(open(f"{args.out}/delta.json"))["step_median"], device=DEV) if args.arm == "stepw" else None
        targs = SimpleNamespace(lr=1e-3 if args.setup == "scratch" else 5e-4, time_budget=0.0, iters=args.iters, k_start=4, k_end=20, batch=16, steps=30,
                                device=str(DEV), seed=args.seed, log_every=100, opt="", rloss=rl, vloss="euclid" if args.arm == "vel_euclid" else "sq", step_err=step_err, huber_delta=args.delta, long_windows=0)
        n_it, secs = ts.train(model, tensors, targs, f"{args.ckpt}/{key}.pt", lambda s: print(key, s, flush=True))
        r.update(train_iters=n_it, train_seconds=secs)
    model.eval()
    r["eval"] = evaluate(model)
    json.dump(r, open(out_path, "w"))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
