"""Learned node-pair error estimator for the momentum-symmetric dual tree (scripts/dual_tree.py), with tail control.
Label per node pair (A, B): y = log max(e_A / (m_B |g|), e_B / (m_A |g|)), e_A = rms over sampled particles of A of
|sum_{j in B} f_ij - (m_B g(r) - G (x_i - COM_A))| (the first-order local expansion), e_B the mirror. Features are exchange-symmetric
(mean and |difference| of per-node terms, quadrupole projected on the line of centres) so both orders of a pair decide alike and
momentum stays exactly conserved. Decision: accept a candidate pair (size/dist < theta_max) when lam * exp(yhat) * m_B |g| <= tol * S_A
and the mirror <= tol * S_B, S = mean |a| of the node's particles from a coarse pass. Tail control: quantile heads (q90, q99) and an
audit controller that samples accepted pairs, measures their true error, and adapts lam so the violation rate stays at delta.

Usage: PYTHONPATH=. python scripts/dual_estimator.py --out results/dual_estimator.json
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from scripts import adaptive_oracle as ao
from scripts import dual_tree as dt
from scripts import kernels

EPS = ao.EPS
KERNEL_FEATS = False


def gmag(dist):
    return kernels.current.f(dist)


def g_and_grad(r):
    return kernels.current.g_grad(r)


def pair_feats(fl, ga, gb):
    r = fl.com[gb] - fl.com[ga]
    dist = r.norm(dim=1)
    rh = r / dist[:, None]
    th = torch.stack([-rh[:, 1], rh[:, 0]], 1)

    def node(g):
        q = fl.q[g]
        qrr = q[:, 0] * rh[:, 0] ** 2 + 2 * q[:, 1] * rh[:, 0] * rh[:, 1] + q[:, 2] * rh[:, 1] ** 2
        qtt = q[:, 0] * th[:, 0] ** 2 + 2 * q[:, 1] * th[:, 0] * th[:, 1] + q[:, 2] * th[:, 1] ** 2
        qrt = q[:, 0] * rh[:, 0] * th[:, 0] + q[:, 1] * (rh[:, 0] * th[:, 1] + rh[:, 1] * th[:, 0]) + q[:, 2] * rh[:, 1] * th[:, 1]
        corr = (qrt.abs() / (qrr.clamp(min=0) * qtt.clamp(min=0) + 1e-30).sqrt()).clamp(max=1)
        return torch.stack([torch.log(fl.count[g].double()), torch.log(fl.size[g] / dist), torch.log((qrr / dist ** 2).clamp(min=1e-14)),
                            torch.log((qtt / dist ** 2).clamp(min=1e-14)), corr], 1)
    fa, fb = node(ga), node(gb)
    mx = torch.maximum(fl.size[ga], fl.size[gb])
    parts = [torch.stack([torch.log(mx / dist), torch.log(dist / EPS)], 1), (fa + fb) / 2, (fa - fb).abs()]
    if KERNEL_FEATS:
        parts.append(kernels.current.local_feats(dist))
    return torch.cat(parts, 1)


def node_scale(fl, a_sorted):
    C = torch.cat([torch.zeros(1, dtype=torch.float64, device=a_sorted.device), torch.cumsum(a_sorted.norm(dim=1), 0)])
    return (C[fl.start + fl.count] - C[fl.start]) / fl.count


@torch.no_grad()
def pair_err(tr, fl, ga, gb, gen, budget=30_000_000):
    dev = ga.device
    P = len(ga)
    cA, cB = fl.count[ga], fl.count[gb]
    ns = (2_000_000 // cB.clamp(min=1)).clamp(4, 48)
    rep = torch.arange(P, device=dev).repeat_interleave(ns)
    off = (torch.rand(len(rep), device=dev, generator=gen, dtype=torch.float64) * cA[rep]).long()
    off = torch.minimum(off, cA[rep] - 1)
    pi = fl.start[ga][rep] + off
    cnt_s, st_s = cB[rep], fl.start[gb][rep]
    S = len(rep)
    fs = torch.zeros(S, 2, dtype=torch.float64, device=dev)
    cs = torch.cumsum(cnt_s, 0)
    i0 = 0
    while i0 < S:
        base = int(cs[i0 - 1]) if i0 else 0
        j0 = min(max(int(torch.searchsorted(cs, torch.tensor(base + budget, device=dev), right=True)), i0 + 1), S)
        c = cnt_s[i0:j0]
        E = int(c.sum())
        r2 = torch.arange(j0 - i0, device=dev).repeat_interleave(c)
        k = torch.arange(E, device=dev) - (torch.cumsum(c, 0) - c)[r2]
        fs[i0:j0].index_add_(0, r2, kernels.current.pair(tr.pos[st_s[i0:j0][r2] + k] - tr.pos[pi[i0:j0]][r2]))
        i0 = j0
    r = fl.com[gb] - fl.com[ga]
    g, grad = g_and_grad(r)
    dx = tr.pos[pi] - fl.com[ga][rep]
    gg = grad[rep] * cB[rep].double()[:, None]
    approx = g[rep] * cB[rep].double()[:, None] - torch.stack([gg[:, 0] * dx[:, 0] + gg[:, 1] * dx[:, 1], gg[:, 2] * dx[:, 0] + gg[:, 3] * dx[:, 1]], 1)
    e2 = ((fs - approx) ** 2).sum(1)
    return (torch.zeros(P, dtype=torch.float64, device=dev).index_add_(0, rep, e2) / ns).sqrt()


def label(tr, fl, ga, gb, gen):
    ea, eb = pair_err(tr, fl, ga, gb, gen), pair_err(tr, fl, gb, ga, gen)
    g = gmag((fl.com[gb] - fl.com[ga]).norm(dim=1))
    rel = torch.maximum(ea / (fl.count[gb] * g), eb / (fl.count[ga] * g))
    return torch.log(rel.clamp(min=1e-6)), ea, eb


def collect_pairs(tr, fl, thetas, cap, n_pairs, n_big, gen):
    dev = tr.pos.device
    ntot = len(fl.count)
    keys = []
    for th in thetas:
        col = []
        dt.dual_accel(tr, fl, th, cap, collect=col)
        ga, gb = torch.cat([c[0] for c in col]), torch.cat([c[1] for c in col])
        m = ga < gb
        keys.append(ga[m] * ntot + gb[m])
    key = torch.unique(torch.cat(keys))
    ga, gb = key // ntot, key % ntot
    big = (fl.count[ga] * fl.count[gb]) > 1_000_000
    bi = big.nonzero().squeeze(1)
    if len(bi) > n_big:
        bi = bi[torch.randperm(len(bi), generator=gen, device=dev)[:n_big]]
    si = (~big).nonzero().squeeze(1)
    if len(si) > n_pairs:
        si = si[torch.randperm(len(si), generator=gen, device=dev)[:n_pairs]]
    sel = torch.cat([bi, si])
    return ga[sel], gb[sel]


class Est(nn.Module):
    def __init__(self, mu, sd, nout, width=128):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.head = 0
        self.net = nn.Sequential(nn.Linear(mu.numel(), width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, nout))

    def forward(self, x):
        return self.net(((x - self.mu) / self.sd).float())[:, self.head].double()


def train(xtr, ytr, xva, yva, taus, dev, epochs, seed):
    torch.manual_seed(seed)
    m = Est(xtr.mean(0).to(dev), (xtr.std(0) + 1e-6).to(dev), 1 + len(taus)).to(dev)
    xt, yt, xv, yv = xtr.to(dev), ytr.to(dev), xva.to(dev), yva.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss(x, y):
        out = m.net(((x - m.mu) / m.sd).float()).double()
        l = ((y - out[:, 0]) ** 2).mean()
        for i, tau in enumerate(taus):
            e = y - out[:, 1 + i]
            l = l + torch.maximum(tau * e, (tau - 1) * e).mean()
        return l

    for ep in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(perm), 2048):
            b = perm[i:i + 2048]
            opt.zero_grad()
            loss(xt[b], yt[b]).backward()
            opt.step()
        sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            with torch.no_grad():
                print("epoch", ep, "train", float(loss(xt, yt)), "val", float(loss(xv, yv)), flush=True)
    return m.eval()


class Head(nn.Module):
    def __init__(self, est, k):
        super().__init__()
        self.est, self.k = est, k

    def forward(self, x):
        return self.est.net(((x - self.est.mu) / self.est.sd).float())[:, self.k].double()


def make_accept(fl, head, lam, tol, fs_node, chunk=2_000_000):
    def fn(ga, gb, acc):
        idx = acc.nonzero().squeeze(1)
        out = torch.zeros_like(acc)
        if len(idx) == 0:
            return out
        for i0 in range(0, len(idx), chunk):
            ic = idx[i0:i0 + chunk]
            a, b = ga[ic], gb[ic]
            with torch.no_grad():
                yh = head(pair_feats(fl, a, b))
            g = gmag((fl.com[b] - fl.com[a]).norm(dim=1))
            ea = lam * torch.exp(yh) * fl.count[b] * g
            eb = lam * torch.exp(yh) * fl.count[a] * g
            ok = ((ea <= tol * fs_node[a]) & (eb <= tol * fs_node[b])) | ((fl.count[a] == 1) & (fl.count[b] == 1))
            out[ic] = ok
        return out
    return fn


def metrics2(a, a_ex):
    d = (a.double() - a_ex).norm(dim=1)
    m = a_ex.norm(dim=1)
    rel = d / m
    rms = m.pow(2).mean().sqrt()
    return {"rel_l2": float(d.norm() / m.norm()), "p99": float(rel.quantile(0.99)), "max": float(rel.max()),
            "abs_p99": float((d / rms).quantile(0.99)), "abs_max": float((d / rms).max())}


def run_est(tr, fl, S, a_ex, cap, head, lam, tol, fs_node, theta_max):
    col = []
    a_s, n_m2l, n_dir = dt.dual_accel(tr, fl, theta_max, cap, accept_fn=make_accept(fl, head, lam, tol, fs_node), collect=col)
    a = torch.empty_like(a_s)
    a[tr.order] = a_s
    N = len(a)
    r = {"cost": (n_m2l + n_dir) / N, "m2l": n_m2l / N, **metrics2(a[S], a_ex), "net_ratio": ao.net_ratio(a)}
    run_est.last_a = a
    return r, col


def audit(tr, fl, col, fs_node, tol, K, gen):
    ga, gb = torch.cat([c[0] for c in col]), torch.cat([c[1] for c in col])
    m = ga < gb
    ga, gb = ga[m], gb[m]
    multi = (fl.count[ga] > 1) | (fl.count[gb] > 1)
    ga, gb = ga[multi], gb[multi]
    if len(ga) > K:
        sel = torch.randperm(len(ga), generator=gen, device=ga.device)[:K]
        ga, gb = ga[sel], gb[sel]
    ea, eb = pair_err(tr, fl, ga, gb, gen), pair_err(tr, fl, gb, ga, gen)
    return float(((ea > tol * fs_node[ga]) | (eb > tol * fs_node[gb])).double().mean())


def audit_e2e(pos, a_orig, K, gen):
    n = pos.shape[0]
    idx = torch.randperm(n, generator=gen, device=pos.device)[:K]
    a_ex = kernels.current.exact_accel(pos, idx, 128)
    return float((a_orig[idx].double() - a_ex).norm() / a_ex.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--train-states", default="uniform,flyby_t0,flyby_t1000")
    ap.add_argument("--test-states", default="flyby_t5000,flyby_t10000")
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--lmax", type=int, default=18)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--theta-max", type=float, default=1.2)
    ap.add_argument("--label-thetas", type=float, nargs="+", default=[0.35, 0.7, 1.2])
    ap.add_argument("--n-pairs", type=int, default=40000)
    ap.add_argument("--n-big", type=int, default=4000)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.9, 0.99])
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--geo-thetas", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.7])
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--audit-k", type=int, default=3000)
    ap.add_argument("--ctrl-tols", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--ctrl-iters", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-targets", type=int, default=20000)
    ap.add_argument("--out", default="results/dual_estimator.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4738)
    states = {}
    data = {}
    for s in args.train_states.split(",") + args.test_states.split(","):
        pos = ao.load_state(s, args, dev)
        tr = ao.Tree(pos, args.lmax)
        fl = dt.Flat(tr)
        ga, gb = collect_pairs(tr, fl, args.label_thetas, args.cap, args.n_pairs, args.n_big, gen)
        t0 = time.time()
        y, ea, eb = label(tr, fl, ga, gb, gen)
        x = pair_feats(fl, ga, gb)
        keep = torch.isfinite(x).all(1) & torch.isfinite(y)
        data[s] = (x[keep].cpu(), y[keep].cpu())
        states[s] = (pos, tr, fl)
        print(s, "pairs", int(keep.sum()), "y range", float(y.min()), float(y.max()), f"label time {time.time() - t0:.1f}s", flush=True)
    tr_s, te_s = args.train_states.split(","), args.test_states.split(",")
    xtr, ytr = torch.cat([data[s][0] for s in tr_s]), torch.cat([data[s][1] for s in tr_s])
    xva, yva = torch.cat([data[s][0] for s in te_s]), torch.cat([data[s][1] for s in te_s])
    est = train(xtr, ytr, xva, yva, args.taus, dev, args.epochs, 4738)
    out = {"args": vars(args), "fit": {}, "eval": {}}
    heads = {"mean": Head(est, 0)}
    for i, tau in enumerate(args.taus):
        heads[f"q{int(tau * 100)}"] = Head(est, 1 + i)
    for s in tr_s + te_s:
        x, y = data[s]
        with torch.no_grad():
            o = {n: h(x.to(dev)).cpu() for n, h in heads.items()}
        ss = ((y - y.mean()) ** 2).sum()
        out["fit"][s] = {"r2_mean": float(1 - ((y - o["mean"]) ** 2).sum() / ss), "n": len(y),
                         **{f"cover_{n}": float((y <= o[n]).double().mean()) for n in heads if n != "mean"}}
        print(s, "fit", json.dumps(out["fit"][s]), flush=True)
    torch.save(est.state_dict(), args.out.replace("results/", "checkpoints/").replace(".json", ".pt"))
    for s in te_s + tr_s:
        pos, tr, fl = states[s]
        N = pos.shape[0]
        S = torch.randperm(N, generator=torch.Generator().manual_seed(4738))[:args.eval_targets].to(dev)
        a_ex = kernels.current.exact_accel(pos, S)
        r = {"geo": [], "est": [], "ctrl": []}
        for th in args.geo_thetas:
            a_s, n_m2l, n_dir = dt.dual_accel(tr, fl, th, args.cap)
            a = torch.empty_like(a_s)
            a[tr.order] = a_s
            row = {"theta": th, "cost": (n_m2l + n_dir) / N, **metrics2(a[S], a_ex)}
            r["geo"].append(row)
            print(s, "geo", json.dumps(row), flush=True)
        a_c, _, _ = dt.dual_accel(tr, fl, args.theta_max, args.cap)
        fs_node = node_scale(fl, a_c)
        for hn, h in heads.items():
            for tol in args.tols:
                row, _ = run_est(tr, fl, S, a_ex, args.cap, h, 1.0, tol, fs_node, args.theta_max)
                row.update({"head": hn, "tol": tol})
                r["est"].append(row)
                print(s, "est", json.dumps(row), flush=True)
        for tol in args.ctrl_tols:
            lam = 1.0
            for it in range(args.ctrl_iters):
                row, col = run_est(tr, fl, S, a_ex, args.cap, heads["mean"], lam, tol, fs_node, args.theta_max)
                viol = audit(tr, fl, col, fs_node, tol, args.audit_k, gen)
                row.update({"head": "mean+audit", "tol": tol, "lam": lam, "violation": viol, "iter": it})
                r["ctrl"].append(row)
                print(s, "ctrl", json.dumps(row), flush=True)
                lam = lam * 1.3 if viol > args.delta else lam / 1.1
        out["eval"][s] = r
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
