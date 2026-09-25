"""Oracle test for adaptive-granularity far field (2D softened gravity, analytic kernel, N=100k, GPU).
Methods, all against exact fp64 all-pairs acceleration on a random target subset:
  mesh    uniform particle mesh (cic, fp64 FFT) + cutoff-4 near field, scripts/gravity_1b.py, grid sweep
  mac     Barnes-Hut quadtree (Morton-sorted, monopole), accept a node when size < theta * dist(target, COM); leaf cap = direct-sum
          threshold (density-adaptive subdivision), theta x cap sweep
  oracle  same tree, but a node is accepted when its TRUE monopole error |F_exact - F_mono| <= eps_node * |F_target| (exact node
          contribution computed per visited pair): the ideal per-cell error estimator, the cost floor a learned estimator could reach
Cost = kernel evaluations per target (accepted-node monopoles + directly summed sources) and wall time; net-force ratio
|sum a| / sum |a| as the momentum diagnostic. Per-(target, node) discrepancy of monopole vs exact is logged for the loosest MAC
run (theta 1.2) to results/adaptive_oracle_cells_<state>.npz as training data for an error estimator.
States: uniform scaled init, and frames of the 100k rotating-cluster flyby (results/flyby_100k.npz truth, record 40 ticks).

Usage: PYTHONPATH=. python scripts/adaptive_oracle.py --out results/adaptive_oracle.json
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import gravity_1b as g1
from model.token_graph import build_radius_graph_cells

EPS = 0.5
FRAMES = {"flyby_t0": 0, "flyby_t1000": 25, "flyby_t5000": 125, "flyby_t10000": 250}


def sync():
    torch.cuda.synchronize()
    return time.time()


def morton(ix, iy, bits):
    k = torch.zeros_like(ix)
    for b in range(bits):
        k |= ((ix >> b) & 1) << (2 * b + 1)
        k |= ((iy >> b) & 1) << (2 * b)
    return k


class Tree:
    def __init__(self, pos, lmax):
        dev = pos.device
        lo = pos.min(0).values
        D = float((pos.max(0).values - lo).max()) * (1 + 1e-6) + 1e-6
        n = 2 ** lmax
        ij = ((pos - lo) / D * n).long().clamp_(0, n - 1)
        key = morton(ij[:, 0], ij[:, 1], lmax)
        order = torch.argsort(key)
        self.order = order
        self.inv = torch.empty_like(order)
        self.inv[order] = torch.arange(len(order), device=dev)
        self.pos = pos[order].double()
        self.key = key[order]
        self.lmax, self.D = lmax, D
        self.lv = []
        for l in range(lmax + 1):
            kl = self.key >> (2 * (lmax - l))
            u, inv, c = torch.unique_consecutive(kl, return_inverse=True, return_counts=True)
            st = torch.cumsum(c, 0) - c
            com = torch.zeros(len(u), 2, dtype=torch.float64, device=dev).index_add_(0, inv, self.pos) / c[:, None]
            dv = self.pos - com[inv]
            q = torch.zeros(len(u), 3, dtype=torch.float64, device=dev).index_add_(
                0, inv, torch.stack([dv[:, 0] ** 2, dv[:, 0] * dv[:, 1], dv[:, 1] ** 2], 1)) / c[:, None]
            self.lv.append(dict(key=u, count=c, start=st, com=com, q=q, size=D / 2 ** l))


def node_exact(tr, tp, ft, start, cnt, budget):
    P = len(ft)
    dev = tp.device
    out = torch.zeros(P, 2, dtype=torch.float64, device=dev)
    if P == 0:
        return out
    cs = torch.cumsum(cnt, 0)
    i = 0
    while i < P:
        base = int(cs[i - 1]) if i else 0
        j = int(torch.searchsorted(cs, torch.tensor(base + budget, device=dev), right=True))
        j = min(max(j, i + 1), P)
        c = cnt[i:j]
        E = int(c.sum())
        rep = torch.arange(j - i, device=dev).repeat_interleave(c)
        off = torch.arange(E, device=dev) - (torch.cumsum(c, 0) - c)[rep]
        d = tr.pos[start[i:j][rep] + off] - tp[ft[i:j]][rep]
        w = ((d ** 2).sum(1) + EPS ** 2) ** -1.5
        out[i:j].index_add_(0, rep, d * w[:, None])
        i = j
    return out


def traverse(tr, ti, cap, mode="mac", theta=0.5, eps_node=None, ftgt=None, log=False, budget=30_000_000):
    dev = tr.pos.device
    M = len(ti)
    tp, tk = tr.pos[ti], tr.key[ti]
    acc = torch.zeros(M, 2, dtype=torch.float64, device=dev)
    n_mono = torch.zeros(M, device=dev)
    n_src = torch.zeros(M, device=dev)
    rows = []
    ft = torch.arange(M, device=dev)
    fn = torch.zeros(M, dtype=torch.long, device=dev)
    for l in range(tr.lmax + 1):
        if len(ft) == 0:
            break
        lv = tr.lv[l]
        cnt, com, start = lv["count"][fn], lv["com"][fn], lv["start"][fn]
        d = com - tp[ft]
        dist = d.norm(dim=1)
        contains = (tk[ft] >> (2 * (tr.lmax - l))) == lv["key"][fn]
        fmono = d * (cnt * (dist ** 2 + EPS ** 2) ** -1.5)[:, None]
        leaf = (cnt <= cap) | (l == tr.lmax)
        fex = None
        if mode == "oracle":
            fex = node_exact(tr, tp, ft, start, cnt, budget)
            accept = (~contains) & ((fex - fmono).norm(dim=1) <= eps_node * ftgt[ft])
        else:
            accept = (~contains) & (lv["size"] < theta * dist)
        if log and accept.any():
            fa = node_exact(tr, tp, ft[accept], start[accept], cnt[accept], budget)
            a_t, a_n = ft[accept], fn[accept]
            rows.append(torch.cat([
                a_t[:, None].double(), torch.full_like(a_t, l)[:, None].double(), cnt[accept][:, None].double(),
                torch.full_like(a_t, lv["size"], dtype=torch.float64)[:, None], dist[accept][:, None], lv["q"][a_n],
                fa.norm(dim=1)[:, None], (fa - fmono[accept]).norm(dim=1)[:, None], ftgt[a_t][:, None]], 1).cpu())
        if accept.any():
            acc.index_add_(0, ft[accept], fmono[accept])
            n_mono.index_add_(0, ft[accept], torch.ones(int(accept.sum()), device=dev))
        direct = (~accept) & leaf
        if direct.any():
            fd = fex[direct] if fex is not None else node_exact(tr, tp, ft[direct], start[direct], cnt[direct], budget)
            acc.index_add_(0, ft[direct], fd)
            n_src.index_add_(0, ft[direct], cnt[direct].float())
        rest = (~accept) & (~leaf)
        if not rest.any():
            break
        rt, rn = ft[rest], fn[rest]
        nxt = tr.lv[l + 1]["key"]
        ck = ((lv["key"][rn].unsqueeze(1) << 2) | torch.arange(4, device=dev)).reshape(-1)
        idx = torch.searchsorted(nxt, ck).clamp_(max=len(nxt) - 1)
        ok = nxt[idx] == ck
        ft, fn = rt.repeat_interleave(4)[ok], idx[ok]
    out = None
    if log and rows:
        out = torch.cat(rows).numpy().astype(np.float32)
    return acc, n_mono, n_src, out


def accel_all(tr, ti, cap, chunk=25000, **kw):
    parts = [traverse(tr, ti[i:i + chunk], cap, **kw) for i in range(0, len(ti), chunk)]
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]))


def exact_accel(pos, idx, chunk=1024):
    src = pos.double()
    out = torch.empty(len(idx), 2, dtype=torch.float64, device=pos.device)
    for i in range(0, len(idx), chunk):
        d = src[None] - src[idx[i:i + chunk]][:, None]
        out[i:i + chunk] = (d * (((d ** 2).sum(-1) + EPS ** 2) ** -1.5)[..., None]).sum(1)
    return out


def metrics(a, a_ex):
    d = (a.double() - a_ex).norm(dim=1)
    m = a_ex.norm(dim=1)
    rel = d / m
    return {"rel_l2": float(d.norm() / m.norm()), "median": float(rel.median()), "p99": float(rel.quantile(0.99)), "max": float(rel.max())}


def net_ratio(a):
    return float(a.double().sum(0).norm() / a.double().norm(dim=1).sum())


def load_state(name, args, dev, cache={}):
    if name == "uniform":
        return g1.init_state(args.bodies, dev, 4738)[0]
    if "truth" not in cache:
        d = np.load(args.npz)
        cache["truth"] = d["truth"]
        print("record", int(d["record"]), "frames", cache["truth"].shape, flush=True)
    return torch.from_numpy(cache["truth"][FRAMES[name]].copy()).to(dev)


def run_state(name, args, dev):
    pos = load_state(name, args, dev)
    N = pos.shape[0]
    gen = torch.Generator().manual_seed(4738)
    S = torch.randperm(N, generator=gen)[:args.eval_targets].to(dev)
    t0 = sync()
    a_ex = exact_accel(pos, S)
    res = {"N": N, "exact_time_s": sync() - t0, "extent": float((pos.max(0).values - pos.min(0).values).max()),
           "median_radius": float((pos - pos.median(0).values).norm(dim=1).median())}
    print(name, "N", N, "extent", res["extent"], "median_radius", res["median_radius"], flush=True)

    res["mesh"] = []
    f = g1.analytic_force(EPS)
    for grid in args.grids:
        g1.far_accel(f, pos, 4.0, grid)
        t0 = sync()
        near = g1.tiled_accel(f, pos, 4.0, 1)
        t1 = sync()
        far = g1.far_accel(f, pos, 4.0, grid)
        t2 = sync()
        a = near + far
        edges = build_radius_graph_cells(pos, 4.0).shape[1]
        r = {"grid": grid, "near_s": t1 - t0, "far_s": t2 - t1, "near_pairs_per_target": edges / N, **metrics(a[S], a_ex), "net_ratio": net_ratio(a)}
        res["mesh"].append(r)
        print(name, "mesh", json.dumps(r), flush=True)

    t0 = sync()
    tr = Tree(pos, args.lmax)
    res["tree_build_s"] = sync() - t0
    ti_all = torch.arange(N, device=dev)
    Slog = S[:args.log_targets]
    ti_log = tr.inv[Slog]
    ftgt = a_ex[:args.log_targets].norm(dim=1)
    ti_S = tr.inv[S]
    res["mac"], mac_cost_log = [], {}
    for cap in args.caps:
        for theta in args.thetas:
            if not res["mac"]:
                accel_all(tr, ti_all[:5000], cap, theta=theta)
            t0 = sync()
            acc, nm, ns = accel_all(tr, ti_all, cap, theta=theta)
            t = sync() - t0
            a = torch.empty_like(acc)
            a[tr.order] = acc
            r = {"theta": theta, "cap": cap, "traverse_s": t, "mean_mono": float(nm.mean()), "mean_direct": float(ns.mean()),
                 "mean_cost": float((nm + ns).mean()), "mean_cost_logsubset": float((nm + ns)[ti_log].mean()),
                 **metrics(a[S], a_ex), "net_ratio": net_ratio(a)}
            res["mac"].append(r)
            print(name, "mac", json.dumps(r), flush=True)
    res["oracle"] = []
    for cap in args.caps:
        for en in args.eps_nodes:
            t0 = sync()
            acc, nm, ns, _ = traverse(tr, ti_log, cap, mode="oracle", eps_node=en, ftgt=ftgt)
            r = {"cap": cap, "eps_node": en, "mean_cost": float((nm + ns).mean()), "mean_mono": float(nm.mean()), "mean_direct": float(ns.mean()),
                 "time_s": sync() - t0, **metrics(acc, a_ex[:args.log_targets])}
            res["oracle"].append(r)
            print(name, "oracle", json.dumps(r), flush=True)
    _, nm, ns, rows = traverse(tr, ti_log, 32, mode="mac", theta=args.log_theta, ftgt=ftgt, log=True)
    np.savez_compressed(f"{args.out_prefix}_cells_{name}.npz", rows=rows,
                        cols=np.array(["target", "level", "count", "size", "dist", "qxx", "qxy", "qyy", "f_exact", "f_err", "f_target"]))
    res["logged_rows"] = int(rows.shape[0])
    print(name, "logged", rows.shape, flush=True)
    del tr
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--states", default="uniform,flyby_t0,flyby_t1000,flyby_t5000,flyby_t10000")
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--lmax", type=int, default=18)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.7])
    ap.add_argument("--caps", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--eps-nodes", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--grids", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--eval-targets", type=int, default=20000)
    ap.add_argument("--log-targets", type=int, default=2000)
    ap.add_argument("--log-theta", type=float, default=1.2)
    ap.add_argument("--out", default="results/adaptive_oracle.json")
    args = ap.parse_args()
    args.out_prefix = args.out[:-5]
    dev = torch.device("cuda")
    out = {"args": {k: v for k, v in vars(args).items()}}
    for name in args.states.split(","):
        out[name] = run_state(name, args, dev)
        json.dump(out, open(args.out, "w"))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
