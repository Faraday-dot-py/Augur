"""Momentum-symmetric hierarchical force (dual-tree, node-node, first-order local expansions), 2D softened gravity, GPU.
Ordered node pairs (A, B) start at (root, root). A pair is accepted when max(size_A, size_B) < theta * |COM_A - COM_B| (a symmetric
function of the unordered pair) and contributes acceleration a_A += m_B g(r), tidal tensor G_A += m_B grad g(r) to A's local
expansion (r = COM_B - COM_A); the mirrored pair (B, A) adds the exactly opposite a_B and the same G_B, so the node-level force is
antisymmetric and the total momentum is conserved to round-off. Unaccepted pairs split the larger (non-leaf) node; two leaves
(count <= cap or finest level) sum all particle pairs directly (each ordered pair adds to its own target; the mirrored pair adds the
exact negative). Local expansions are pushed to children (L2L, exact for linear fields, sum of child forces = parent force) and
evaluated at each particle: a_p = a_leaf + G_leaf (x_p - COM_leaf).

Usage: PYTHONPATH=. python scripts/dual_tree.py --out results/dual_tree.json
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import adaptive_oracle as ao

EPS = ao.EPS


class Flat:
    def __init__(self, tr):
        dev = tr.pos.device
        lv = tr.lv
        lens = [len(x["key"]) for x in lv]
        self.off = np.concatenate([[0], np.cumsum(lens)])
        self.count = torch.cat([x["count"] for x in lv])
        self.start = torch.cat([x["start"] for x in lv])
        self.com = torch.cat([x["com"] for x in lv])
        self.level = torch.cat([torch.full((n,), l, device=dev, dtype=torch.long) for l, n in enumerate(lens)])
        self.size = torch.tensor([tr.D / 2 ** l for l in range(tr.lmax + 1)], dtype=torch.float64, device=dev)[self.level]
        self.child = torch.full((len(self.count), 4), -1, dtype=torch.long, device=dev)
        for l in range(tr.lmax):
            nxt = lv[l + 1]["key"]
            ck = (lv[l]["key"].unsqueeze(1) << 2) | torch.arange(4, device=dev)
            idx = torch.searchsorted(nxt, ck.reshape(-1)).clamp_(max=len(nxt) - 1)
            ok = (nxt[idx] == ck.reshape(-1)).view(-1, 4)
            self.child[self.off[l]:self.off[l + 1]] = torch.where(ok, idx.view(-1, 4) + int(self.off[l + 1]), torch.full_like(idx.view(-1, 4), -1))
        newnode = torch.cat([torch.ones(1, dtype=torch.long, device=dev), (tr.key[1:] != tr.key[:-1]).long()])
        self.pnode = torch.cumsum(newnode, 0) - 1 + int(self.off[tr.lmax])
        self.lmax = tr.lmax


def direct(tr, fl, ga, gb, a, budget=30_000_000):
    if len(ga) == 0:
        return 0
    ca, cb = fl.count[ga], fl.count[gb]
    npair = ca * cb
    cs = torch.cumsum(npair, 0)
    i0, P = 0, len(ga)
    dev = ga.device
    while i0 < P:
        base = int(cs[i0 - 1]) if i0 else 0
        j0 = int(torch.searchsorted(cs, torch.tensor(base + budget, device=dev), right=True))
        j0 = min(max(j0, i0 + 1), P)
        npr = npair[i0:j0]
        E = int(npr.sum())
        rep = torch.arange(j0 - i0, device=dev).repeat_interleave(npr)
        k = torch.arange(E, device=dev) - (torch.cumsum(npr, 0) - npr)[rep]
        cbb = cb[i0:j0][rep]
        pi = fl.start[ga[i0:j0]][rep] + k // cbb
        pj = fl.start[gb[i0:j0]][rep] + k % cbb
        d = tr.pos[pj] - tr.pos[pi]
        w = ((d ** 2).sum(1) + EPS ** 2) ** -1.5
        a.index_add_(0, pi, d * w[:, None])
        i0 = j0
    return int(cs[-1])


@torch.no_grad()
def dual_accel(tr, fl, theta, cap):
    dev = tr.pos.device
    n_nodes = len(fl.count)
    a_loc = torch.zeros(n_nodes, 2, dtype=torch.float64, device=dev)
    g_loc = torch.zeros(n_nodes, 4, dtype=torch.float64, device=dev)
    a_dir = torch.zeros(len(tr.pos), 2, dtype=torch.float64, device=dev)
    ga = torch.zeros(1, dtype=torch.long, device=dev)
    gb = torch.zeros(1, dtype=torch.long, device=dev)
    n_m2l, n_direct = 0, 0
    while len(ga) > 0:
        ca, cb = fl.count[ga], fl.count[gb]
        r = fl.com[gb] - fl.com[ga]
        dist = r.norm(dim=1)
        sa, sb = fl.size[ga], fl.size[gb]
        acc = (torch.maximum(sa, sb) < theta * dist) & (ga != gb)
        if acc.any():
            r_, cb_, ga_ = r[acc], cb[acc].double(), ga[acc]
            s2 = (r_ ** 2).sum(1) + EPS ** 2
            i3, i5 = s2 ** -1.5, s2 ** -2.5
            a_loc.index_add_(0, ga_, r_ * (cb_ * i3)[:, None])
            grad = torch.stack([i3 - 3 * r_[:, 0] ** 2 * i5, -3 * r_[:, 0] * r_[:, 1] * i5, -3 * r_[:, 0] * r_[:, 1] * i5, i3 - 3 * r_[:, 1] ** 2 * i5], 1)
            g_loc.index_add_(0, ga_, grad * cb_[:, None])
            n_m2l += int(acc.sum())
        rest = ~acc
        leaf_a = (ca <= cap) | (fl.level[ga] == fl.lmax)
        leaf_b = (cb <= cap) | (fl.level[gb] == fl.lmax)
        dr = rest & leaf_a & leaf_b
        if dr.any():
            n_direct += direct(tr, fl, ga[dr], gb[dr], a_dir)
        sp_a = rest & ~leaf_a & ((sa >= sb) | leaf_b)
        sp_b = rest & ~leaf_b & ((sb >= sa) | leaf_a)
        sp = sp_a | sp_b
        if not sp.any():
            break
        ga, gb, sp_a, sp_b = ga[sp], gb[sp], sp_a[sp], sp_b[sp]
        ea = torch.where(sp_a[:, None], fl.child[ga], torch.stack([ga, *[torch.full_like(ga, -1)] * 3], 1))
        eb = torch.where(sp_b[:, None], fl.child[gb], torch.stack([gb, *[torch.full_like(gb, -1)] * 3], 1))
        pa = ea.unsqueeze(2).expand(-1, 4, 4).reshape(-1)
        pb = eb.unsqueeze(1).expand(-1, 4, 4).reshape(-1)
        ok = (pa >= 0) & (pb >= 0)
        ga, gb = pa[ok], pb[ok]
    for l in range(tr.lmax):
        p = torch.arange(int(fl.off[l]), int(fl.off[l + 1]), device=dev)
        ch = fl.child[p]
        v = ch >= 0
        pp = p[:, None].expand(-1, 4)[v]
        cc = ch[v]
        dcom = fl.com[cc] - fl.com[pp]
        gp = g_loc[pp]
        a_loc.index_add_(0, cc, a_loc[pp] - torch.stack([gp[:, 0] * dcom[:, 0] + gp[:, 1] * dcom[:, 1], gp[:, 2] * dcom[:, 0] + gp[:, 3] * dcom[:, 1]], 1))
        g_loc.index_add_(0, cc, gp)
    dx = tr.pos - fl.com[fl.pnode]
    gl = g_loc[fl.pnode]
    a = a_dir + a_loc[fl.pnode] - torch.stack([gl[:, 0] * dx[:, 0] + gl[:, 1] * dx[:, 1], gl[:, 2] * dx[:, 0] + gl[:, 3] * dx[:, 1]], 1)
    return a, n_m2l, n_direct


def sync():
    torch.cuda.synchronize()
    return time.time()


def run_state(name, args, dev):
    pos = ao.load_state(name, args, dev)
    N = pos.shape[0]
    S = torch.randperm(N, generator=torch.Generator().manual_seed(4738))[:args.eval_targets].to(dev)
    a_ex = ao.exact_accel(pos, S)
    tr = ao.Tree(pos, args.lmax)
    t0 = sync()
    fl = Flat(tr)
    res = {"N": N, "flat_build_s": sync() - t0, "nodes": len(fl.count), "dual": []}
    base = json.load(open(args.base))[name] if args.base else None
    for cap in args.caps:
        for theta in args.thetas:
            if not res["dual"]:
                dual_accel(tr, fl, theta, cap)
            t0 = sync()
            a_s, n_m2l, n_dir = dual_accel(tr, fl, theta, cap)
            t = sync() - t0
            a = torch.empty_like(a_s)
            a[tr.order] = a_s
            r = {"theta": theta, "cap": cap, "time_s": t, "m2l_pairs_per_particle": n_m2l / N, "direct_pairs_per_particle": n_dir / N,
                 "cost_per_particle": (n_m2l + n_dir) / N, **ao.metrics(a[S], a_ex), "net_ratio": ao.net_ratio(a)}
            res["dual"].append(r)
            print(name, "dual", json.dumps(r), flush=True)
    del tr
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--base", default="results/adaptive_oracle.json")
    ap.add_argument("--states", default="uniform,flyby_t0,flyby_t1000,flyby_t5000,flyby_t10000")
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--lmax", type=int, default=18)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.25, 0.35, 0.5, 0.7])
    ap.add_argument("--caps", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--eval-targets", type=int, default=20000)
    ap.add_argument("--out", default="results/dual_tree.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = {"args": vars(args)}
    for s in args.states.split(","):
        out[s] = run_state(s, args, dev)
        json.dump(out, open(args.out, "w"))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
