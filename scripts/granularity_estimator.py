"""Learned error estimator for the adaptive far field (follows scripts/adaptive_oracle.py; needs its results/adaptive_oracle_cells_*.npz).
Trains a small MLP on logged (target, node) rows to predict y = log(|F_exact - F_mono| / |F_mono|) of a node's monopole from
kernel-agnostic node features (size/dist, normalised quadrupole trace + anisotropy, count, dist/eps, size/eps): a mean head (MSE)
and an upper-quantile head (pinball, tau). Then runs the tree with the estimator deciding accept-vs-refine:
accept a node when exp(yhat) * |F_mono| <= tol * |F_target_scale| (scale from a coarse theta=1.2 pass) and size/dist < theta_max
(a hard geometric floor), sweeping tol and head, and compares kernel evals per target at matched error with the geometric MAC
and the oracle rows of results/adaptive_oracle.json. Train states and held-out states are separate.

Usage: PYTHONPATH=. python scripts/granularity_estimator.py --out results/granularity_estimator.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from scripts import adaptive_oracle as ao

EPS = ao.EPS


def load_rows(name, prefix):
    r = torch.from_numpy(np.load(f"{prefix}_cells_{name}.npz")["rows"]).double()
    cnt, size, dist = r[:, 2], r[:, 3], r[:, 4]
    q = r[:, 5:8]
    fm = cnt * dist * (dist ** 2 + EPS ** 2) ** -1.5
    y = torch.log((r[:, 9] / fm).clamp(min=1e-9))
    keep = (cnt > 1) & torch.isfinite(y)
    x = ao.feats(cnt[keep], size[keep], dist[keep], q[keep])
    return x, y[keep]


class Est(nn.Module):
    def __init__(self, mu, sd, head, width=128):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.head = head
        self.net = nn.Sequential(nn.Linear(mu.numel(), width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 2))

    def forward(self, x):
        return self.net(((x - self.mu) / self.sd).float())[:, self.head].double()


def train(xtr, ytr, xva, yva, tau, dev, epochs, seed):
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    both = Est(mu.to(dev), sd.to(dev), 0).to(dev)
    xt, yt, xv, yv = xtr.to(dev), ytr.to(dev), xva.to(dev), yva.to(dev)
    opt = torch.optim.Adam(both.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss(x, y):
        out = both.net(((x - both.mu) / both.sd).float()).double()
        e = y - out[:, 1]
        return ((y - out[:, 0]) ** 2).mean() + torch.maximum(tau * e, (tau - 1) * e).mean()

    for ep in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss(xt[b], yt[b]).backward()
            opt.step()
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            with torch.no_grad():
                print("epoch", ep, "train", float(loss(xt, yt)), "val", float(loss(xv, yv)), flush=True)
    mean_head, q_head = Est(mu.to(dev), sd.to(dev), 0).to(dev), Est(mu.to(dev), sd.to(dev), 1).to(dev)
    q_head.net.load_state_dict(both.net.state_dict())
    mean_head.net.load_state_dict(both.net.state_dict())
    return mean_head.eval(), q_head.eval()


def fit_quality(mean_head, q_head, x, y, dev):
    with torch.no_grad():
        m, q = mean_head(x.to(dev)).cpu(), q_head(x.to(dev)).cpu()
    ss = ((y - y.mean()) ** 2).sum()
    geo = x[:, 1] + 2 * x[:, 0]
    return {"r2_mean": float(1 - ((y - m) ** 2).sum() / ss), "mae_mean": float((y - m).abs().mean()),
            "q_coverage": float((y <= q).double().mean()), "n": int(len(y)),
            "geo_only_r2": float(1 - ((y - (geo + (y - geo).mean())) ** 2).sum() / ss)}


def cost_at(rows, target):
    pts = sorted((r["rel_l2"], r["mean_cost"]) for r in rows)
    e, c = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    if not (e.min() <= np.log(target) <= e.max()):
        return None
    return float(np.exp(np.interp(np.log(target), e, c)))


def evaluate(name, args, dev, heads, base, split):
    pos = ao.load_state(name, args, dev)
    N = pos.shape[0]
    S = torch.randperm(N, generator=torch.Generator().manual_seed(4738))[:args.eval_targets].to(dev)
    a_ex = ao.exact_accel(pos, S)
    tr = ao.Tree(pos, args.lmax)
    ti = tr.inv[S]
    res = {"split": split, "est": []}
    for cap in args.caps:
        acc_c, _, _ = ao.accel_all(tr, ti, cap, theta=args.theta_max)
        fscale = acc_c.norm(dim=1)
        for hname, head in heads.items():
            for tol in args.tols:
                acc, nm, ns = ao.accel_all(tr, ti, cap, mode="est", theta=args.theta_max, eps_node=tol, est=head, fscale=fscale)
                r = {"head": hname, "cap": cap, "tol": tol, "mean_cost": float((nm + ns).mean()), "mean_mono": float(nm.mean()), **ao.metrics(acc, a_ex)}
                res["est"].append(r)
                print(name, "est", json.dumps(r), flush=True)
    b = base[name]
    res["matched"] = {}
    for tgt in (0.02, 0.01, 0.005):
        row = {"mac_cap8": cost_at([x for x in b["mac"] if x["cap"] == 8], tgt), "oracle_cap8": cost_at([x for x in b["oracle"] if x["cap"] == 8], tgt)}
        for hname in heads:
            row[f"est_{hname}_cap8"] = cost_at([x for x in res["est"] if x["head"] == hname and x["cap"] == 8], tgt)
        res["matched"][str(tgt)] = row
    print(name, "matched", json.dumps(res["matched"]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--prefix", default="results/adaptive_oracle")
    ap.add_argument("--base", default="results/adaptive_oracle.json")
    ap.add_argument("--train-states", default="uniform,flyby_t0,flyby_t1000")
    ap.add_argument("--test-states", default="flyby_t5000,flyby_t10000")
    ap.add_argument("--lmax", type=int, default=18)
    ap.add_argument("--caps", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--theta-max", type=float, default=1.2)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-targets", type=int, default=20000)
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--out", default="results/granularity_estimator.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    tr_states, te_states = args.train_states.split(","), args.test_states.split(",")
    data = {s: load_rows(s, args.prefix) for s in tr_states + te_states}
    xtr = torch.cat([data[s][0] for s in tr_states])
    ytr = torch.cat([data[s][1] for s in tr_states])
    xva = torch.cat([data[s][0] for s in te_states])
    yva = torch.cat([data[s][1] for s in te_states])
    print("train rows", len(xtr), "held-out rows", len(xva), "y range", float(ytr.min()), float(ytr.max()), flush=True)
    mean_head, q_head = train(xtr, ytr, xva, yva, args.tau, dev, args.epochs, 4738)
    out = {"args": vars(args), "fit": {}}
    for s in tr_states + te_states:
        out["fit"][s] = fit_quality(mean_head, q_head, data[s][0], data[s][1], dev)
        print(s, "fit", json.dumps(out["fit"][s]), flush=True)
    torch.save({"mean": mean_head.state_dict(), "q": q_head.state_dict(), "args": vars(args)}, args.out.replace("results/", "checkpoints/").replace(".json", ".pt"))
    base = json.load(open(args.base))
    heads = {"mean": mean_head, f"q{int(args.tau * 100)}": q_head}
    out["eval"] = {}
    for s in te_states + tr_states:
        out["eval"][s] = evaluate(s, args, dev, heads, base, "test" if s in te_states else "train")
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
