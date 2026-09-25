"""Instrumented copy of the ALF force call (scripts/dual_tree.py dual_accel + adaptive_force.py est/geo path) with per-stage
synchronised timers; same arithmetic as the originals (used for the breakdown only, totals come from AdaptiveForce itself).
"""
import time
from collections import defaultdict

import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import kernels


class Tm:
    def __init__(self):
        self.d = defaultdict(float)

    def start(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def mark(self, k):
        torch.cuda.synchronize()
        n = time.perf_counter()
        self.d[k] += n - self.t
        self.t = n


@torch.no_grad()
def dual_accel_timed(tr, fl, theta, cap, accept_fn, T):
    dev = tr.pos.device
    n_nodes = len(fl.count)
    a_loc = torch.zeros(n_nodes, 2, dtype=torch.float64, device=dev)
    g_loc = torch.zeros(n_nodes, 4, dtype=torch.float64, device=dev)
    a_dir = torch.zeros(len(tr.pos), 2, dtype=torch.float64, device=dev)
    ga = torch.zeros(1, dtype=torch.long, device=dev)
    gb = torch.zeros(1, dtype=torch.long, device=dev)
    n_m2l, n_direct, iters, maxfront = 0, 0, 0, 0
    T.mark("alloc")
    while len(ga) > 0:
        iters += 1
        maxfront = max(maxfront, len(ga))
        ca, cb = fl.count[ga], fl.count[gb]
        r = fl.com[gb] - fl.com[ga]
        dist = r.norm(dim=1)
        sa, sb = fl.size[ga], fl.size[gb]
        acc = (torch.maximum(sa, sb) < theta * dist) & (ga != gb)
        T.mark("geom")
        if accept_fn is not None:
            acc = accept_fn(ga, gb, acc)
            T.mark("estimator")
        if acc.any():
            r_, cb_, ga_ = r[acc], cb[acc].double(), ga[acc]
            g_, grad = kernels.current.g_grad(r_)
            a_loc.index_add_(0, ga_, g_ * cb_[:, None])
            g_loc.index_add_(0, ga_, grad * cb_[:, None])
            n_m2l += int(acc.sum())
        T.mark("m2l")
        rest = ~acc
        leaf_a = (ca <= cap) | (fl.level[ga] == fl.lmax)
        leaf_b = (cb <= cap) | (fl.level[gb] == fl.lmax)
        dr = rest & leaf_a & leaf_b
        if dr.any():
            n_direct += dt.direct(tr, fl, ga[dr], gb[dr], a_dir)
        T.mark("direct")
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
        T.mark("split")
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
    T.mark("l2l")
    dx = tr.pos - fl.com[fl.pnode]
    gl = g_loc[fl.pnode]
    a = a_dir + a_loc[fl.pnode] - torch.stack([gl[:, 0] * dx[:, 0] + gl[:, 1] * dx[:, 1], gl[:, 2] * dx[:, 0] + gl[:, 3] * dx[:, 1]], 1)
    T.mark("eval")
    return a, n_m2l, n_direct, {"iters": iters, "max_frontier": maxfront, "nodes": n_nodes}


def timed_call(pos, af, T, mode):
    """One ALF force call; mode 'est' (steady state, node scale from af.prev_a) or 'geo' (theta = af.theta)."""
    kernels.current = af.kernel
    T.start()
    tr = ao.Tree(pos, af.lmax)
    T.mark("tree")
    fl = dt.Flat(tr)
    T.mark("flat")
    if mode == "est":
        fs = de.node_scale(fl, af.prev_a[tr.order])
        T.mark("node_scale")
        a_s, m2l, dirn, info = dual_accel_timed(tr, fl, af.theta_max, af.cap, de.make_accept(fl, af.head, af.lam, af.tol, fs), T)
    else:
        a_s, m2l, dirn, info = dual_accel_timed(tr, fl, af.theta, af.cap, None, T)
    a = torch.empty_like(a_s)
    a[tr.order] = a_s
    T.mark("scatter")
    af.prev_a = a
    info.update({"cost": (m2l + dirn) / pos.shape[0], "m2l": m2l / pos.shape[0], "direct": dirn / pos.shape[0]})
    return a, info
