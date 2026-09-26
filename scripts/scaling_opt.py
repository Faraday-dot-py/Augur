"""Semantics-preserving speedups of the ALF force call (copies of scripts/dual_tree.py, adaptive_oracle.Tree, dual_estimator.py paths).
Same node set, same accept rule, same estimator; differences are float round-off / atomic-add order only.
Flags: fast_tree (bottom-up merge tree, no per-level atomics into few nodes, no searchsorted child map), sync-light traversal (always),
analytic_grad (closed-form g_grad for the softened-gravity kernel, kernel-specific), compile (torch.compile of pair features / analytic pair force).
"""
import time
from types import SimpleNamespace

import torch

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import kernels
from scripts.adaptive_force import AdaptiveForce

EPS = ao.EPS


def spread(x):
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    return (x | (x << 1)) & 0x5555555555555555


class OTree:
    """Bottom-up tree: leaf level by segment reduction, coarser levels by exact parallel-axis merge of children. Produces the Flat fields."""

    def __init__(self, pos, lmax):
        dev = pos.device
        lo = pos.min(0).values
        D = float((pos.max(0).values - lo).max()) * (1 + 1e-6) + 1e-6
        n = 2 ** lmax
        ij = ((pos - lo) / D * n).long().clamp_(0, n - 1)
        key = (spread(ij[:, 0]) << 1) | spread(ij[:, 1])
        order = torch.argsort(key)
        self.order = order
        self.pos = pos[order].double()
        self.key = key[order]
        self.lmax, self.D = lmax, D
        N = len(order)
        flag = self.key[1:] != self.key[:-1]
        st = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), flag.nonzero().squeeze(1) + 1])
        cnt = torch.diff(torch.cat([st, torch.tensor([N], device=dev)]))
        seg = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(flag, 0)])
        com = torch.zeros(len(st), 2, dtype=torch.float64, device=dev).index_add_(0, seg, self.pos) / cnt[:, None]
        dv = self.pos - com[seg]
        q = torch.zeros(len(st), 3, dtype=torch.float64, device=dev).index_add_(
            0, seg, torch.stack([dv[:, 0] ** 2, dv[:, 0] * dv[:, 1], dv[:, 1] ** 2], 1)) / cnt[:, None]
        self.seg = seg
        lv = [dict(key=self.key[st], count=cnt, start=st, com=com, q=q)]
        pids = []
        for l in range(lmax - 1, -1, -1):
            c = lv[-1]
            pk = c["key"] >> 2
            pf = pk[1:] != pk[:-1]
            pst = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), pf.nonzero().squeeze(1) + 1])
            pid = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(pf, 0)])
            n_p = len(pst)
            pc = torch.zeros(n_p, dtype=torch.long, device=dev).index_add_(0, pid, c["count"])
            cd = c["count"].double()
            pcom = torch.zeros(n_p, 2, dtype=torch.float64, device=dev).index_add_(0, pid, c["com"] * cd[:, None]) / pc[:, None]
            d = c["com"] - pcom[pid]
            m = torch.stack([c["q"][:, 0] + d[:, 0] ** 2, c["q"][:, 1] + d[:, 0] * d[:, 1], c["q"][:, 2] + d[:, 1] ** 2], 1) * cd[:, None]
            pq = torch.zeros(n_p, 3, dtype=torch.float64, device=dev).index_add_(0, pid, m) / pc[:, None]
            pids.append(pid)
            lv.append(dict(key=pk[pst], count=pc, start=c["start"][pst], com=pcom, q=pq))
        self.lv = lv[::-1]
        self.pids = pids[::-1]


def oflat(tr, cap):
    lv, dev = tr.lv, tr.pos.device
    lens = [len(x["key"]) for x in lv]
    fl = SimpleNamespace()
    import numpy as np
    fl.off = np.concatenate([[0], np.cumsum(lens)])
    fl.count = torch.cat([x["count"] for x in lv])
    fl.q = torch.cat([x["q"] for x in lv])
    fl.start = torch.cat([x["start"] for x in lv])
    fl.com = torch.cat([x["com"] for x in lv])
    fl.level = torch.cat([torch.full((n,), l, device=dev, dtype=torch.long) for l, n in enumerate(lens)])
    fl.size = torch.tensor([tr.D / 2 ** l for l in range(tr.lmax + 1)], dtype=torch.float64, device=dev)[fl.level]
    total = int(fl.off[-1])
    fl.child = torch.full((total, 4), -1, dtype=torch.long, device=dev)
    fl.parent = torch.full((total,), -1, dtype=torch.long, device=dev)
    for l in range(tr.lmax):
        pid = tr.pids[l]
        slot = lv[l + 1]["key"] & 3
        cidx = torch.arange(lens[l + 1], device=dev) + int(fl.off[l + 1])
        fl.child[pid + int(fl.off[l]), slot] = cidx
        fl.parent[cidx] = pid + int(fl.off[l])
    fl.pnode = tr.seg + int(fl.off[tr.lmax])
    fl.lmax = tr.lmax
    return augment(fl, cap)


def augment(fl, cap):
    dev = fl.count.device
    if not hasattr(fl, "parent"):
        total = len(fl.count)
        fl.parent = torch.full((total,), -1, dtype=torch.long, device=dev)
        v = fl.child >= 0
        src = torch.arange(total, device=dev)[:, None].expand(-1, 4)[v]
        fl.parent[fl.child[v]] = src
    fl.cntd = fl.count.double()
    fl.leaf = (fl.count <= cap) | (fl.level == fl.lmax)
    return fl


def gg_analytic(r, eps=EPS):
    rn = r.norm(dim=1).clamp(min=1e-9)
    s = rn * rn + eps * eps
    fx = rn * s ** -1.5
    fp = s ** -1.5 - 3 * rn * rn * s ** -2.5
    rh = r / rn[:, None]
    a = fx / rn
    xx, xy, yy = rh[:, 0] ** 2, rh[:, 0] * rh[:, 1], rh[:, 1] ** 2
    return rh * fx[:, None], torch.stack([a * (1 - xx) + fp * xx, (fp - a) * xy, (fp - a) * xy, a * (1 - yy) + fp * yy], 1)


def _pair_analytic(d, eps=EPS):
    dn = d.norm(dim=1).clamp(min=1e-9)
    return d * ((dn * (dn * dn + eps * eps) ** -1.5) / dn)[:, None]


_compiled = {}


def get_compiled(name):
    if name not in _compiled:
        fn = {"pair_feats": de.pair_feats, "pair_analytic": _pair_analytic}[name]
        compiled = torch.compile(fn, dynamic=True)
        state = {"failed": False}

        def wrapped(*a, **kw):
            if not state["failed"]:
                try:
                    return compiled(*a, **kw)
                except Exception:
                    state["failed"] = True
            return fn(*a, **kw)

        _compiled[name] = wrapped
    return _compiled[name]


class Opts:
    def __init__(self, fast_tree=True, analytic_grad=False, compile=False, chunk=1 << 22):
        self.fast_tree, self.analytic_grad, self.compile, self.chunk = fast_tree, analytic_grad, compile, chunk


def direct_opt(tr, fl, ga, gb, a, o, budget=30_000_000):
    P = len(ga)
    if P == 0:
        return 0
    ca, cb = fl.count[ga], fl.count[gb]
    npair = ca * cb
    cs = torch.cumsum(npair, 0)
    total = int(cs[-1])
    dev = ga.device
    pair = get_compiled("pair_analytic") if o.compile else kernels.current.pair
    i0 = 0
    while i0 < P:
        if total <= budget and i0 == 0:
            j0, E = P, total
            npr, csl = npair, cs
        else:
            base = int(cs[i0 - 1]) if i0 else 0
            j0 = min(max(int(torch.searchsorted(cs, torch.tensor(base + budget, device=dev), right=True)), i0 + 1), P)
            npr = npair[i0:j0]
            csl = torch.cumsum(npr, 0)
            E = int(csl[-1])
        rep = torch.arange(j0 - i0, device=dev).repeat_interleave(npr, output_size=E)
        k = torch.arange(E, device=dev) - (csl - npr)[rep]
        cbb = cb[i0:j0][rep]
        pi = fl.start[ga[i0:j0]][rep] + k // cbb
        pj = fl.start[gb[i0:j0]][rep] + k % cbb
        a.index_add_(0, pi, pair(tr.pos[pj] - tr.pos[pi]))
        i0 = j0
    return total


class Est:
    def __init__(self, fl, head, lam, tol, fs, o):
        self.fl, self.head, self.lam, self.tol, self.fs, self.o = fl, head, lam, tol, fs, o
        self.pf = get_compiled("pair_feats") if o.compile else de.pair_feats

    @torch.no_grad()
    def __call__(self, a, b):
        fl, out = self.fl, []
        for i in range(0, len(a), self.o.chunk):
            ai, bi = a[i:i + self.o.chunk], b[i:i + self.o.chunk]
            yh = self.head(self.pf(fl, ai, bi))
            g = de.gmag((fl.com[bi] - fl.com[ai]).norm(dim=1))
            ea = self.lam * torch.exp(yh) * fl.cntd[bi] * g
            eb = self.lam * torch.exp(yh) * fl.cntd[ai] * g
            out.append(((ea <= self.tol * self.fs[ai]) & (eb <= self.tol * self.fs[bi])) | ((fl.count[ai] == 1) & (fl.count[bi] == 1)))
        return torch.cat(out)


@torch.no_grad()
def dual_accel_opt(tr, fl, theta, cap, o, est=None, T=None):
    dev = tr.pos.device
    mark = T.mark if T is not None else (lambda k: None)
    loc = torch.zeros(len(fl.count), 6, dtype=torch.float64, device=dev)
    a_dir = torch.zeros(len(tr.pos), 2, dtype=torch.float64, device=dev)
    gg = gg_analytic if o.analytic_grad else kernels.current.g_grad
    ga = torch.zeros(1, dtype=torch.long, device=dev)
    gb = torch.zeros(1, dtype=torch.long, device=dev)
    n_m2l = n_direct = 0
    mark("alloc")
    while len(ga) > 0:
        r = fl.com[gb] - fl.com[ga]
        dist = r.norm(dim=1)
        sa, sb = fl.size[ga], fl.size[gb]
        cand = (torch.maximum(sa, sb) < theta * dist) & (ga != gb)
        ia = cand.nonzero().squeeze(1)
        mark("geom")
        if est is not None and len(ia):
            ia = ia[est(ga[ia], gb[ia])]
            mark("estimator")
        if len(ia):
            ga_ = ga[ia]
            g_, grad = gg(r[ia])
            loc.index_add_(0, ga_, torch.cat([g_, grad], 1) * fl.cntd[gb[ia]][:, None])
            n_m2l += len(ia)
        rest = torch.ones(len(ga), dtype=torch.bool, device=dev)
        rest[ia] = False
        mark("m2l")
        leaf_a, leaf_b = fl.leaf[ga], fl.leaf[gb]
        idr = (rest & leaf_a & leaf_b).nonzero().squeeze(1)
        if len(idr):
            n_direct += direct_opt(tr, fl, ga[idr], gb[idr], a_dir, o)
        mark("direct")
        sp_a = rest & ~leaf_a & ((sa >= sb) | leaf_b)
        sp_b = rest & ~leaf_b & ((sb >= sa) | leaf_a)
        isp = (sp_a | sp_b).nonzero().squeeze(1)
        if len(isp) == 0:
            break
        ga, gb, sp_a, sp_b = ga[isp], gb[isp], sp_a[isp], sp_b[isp]
        ea = torch.where(sp_a[:, None], fl.child[ga], torch.stack([ga, *[torch.full_like(ga, -1)] * 3], 1))
        eb = torch.where(sp_b[:, None], fl.child[gb], torch.stack([gb, *[torch.full_like(gb, -1)] * 3], 1))
        idx = ((ea >= 0).unsqueeze(2) & (eb >= 0).unsqueeze(1)).nonzero()
        ga, gb = ea[idx[:, 0], idx[:, 1]], eb[idx[:, 0], idx[:, 2]]
        mark("split")
    for l in range(tr.lmax):
        s0, s1, s2 = int(fl.off[l + 1]), int(fl.off[l + 1]), int(fl.off[l + 2])
        cc = torch.arange(s1, s2, device=dev)
        pp = fl.parent[s1:s2]
        dcom = fl.com[cc] - fl.com[pp]
        gp = loc[pp, 2:]
        loc[s1:s2, :2] += loc[pp, :2] - torch.stack([gp[:, 0] * dcom[:, 0] + gp[:, 1] * dcom[:, 1], gp[:, 2] * dcom[:, 0] + gp[:, 3] * dcom[:, 1]], 1)
        loc[s1:s2, 2:] += gp
    mark("l2l")
    dx = tr.pos - fl.com[fl.pnode]
    gl = loc[fl.pnode, 2:]
    a = a_dir + loc[fl.pnode, :2] - torch.stack([gl[:, 0] * dx[:, 0] + gl[:, 1] * dx[:, 1], gl[:, 2] * dx[:, 0] + gl[:, 3] * dx[:, 1]], 1)
    mark("eval")
    return a, n_m2l, n_direct


def build(pos, lmax, cap, o, T=None):
    if o.fast_tree:
        tr = OTree(pos, lmax)
        if T:
            T.mark("tree")
        fl = oflat(tr, cap)
        if T:
            T.mark("flat")
    else:
        tr = ao.Tree(pos, lmax)
        if T:
            T.mark("tree")
        fl = augment(dt.Flat(tr), cap)
        if T:
            T.mark("flat")
    return tr, fl


class OptForce(AdaptiveForce):
    def __init__(self, *a, opts=None, fast_audit=False, **kw):
        super().__init__(*a, **kw)
        self.o = opts or Opts()
        self.fast_audit = fast_audit
        self.T = None

    def __call__(self, pos):
        kernels.current = self.kernel
        T = self.T
        if T:
            T.start()
        t0 = time.perf_counter() if not T else None
        tr, fl = build(pos, self.lmax, self.cap, self.o, T)
        if self.now == "est":
            if self.prev_a is not None:
                fs = de.node_scale(fl, self.prev_a[tr.order])
            else:
                a_c, _, _ = dual_accel_opt(tr, fl, self.theta_max, self.cap, self.o)
                fs = de.node_scale(fl, a_c)
            if T:
                T.mark("node_scale")
            a_s, m2l, dirn = dual_accel_opt(tr, fl, self.theta_max, self.cap, self.o, est=Est(fl, self.head, self.lam, self.tol, fs, self.o), T=T)
        else:
            a_s, m2l, dirn = dual_accel_opt(tr, fl, self.theta, self.cap, self.o, T=T)
        a = torch.empty_like(a_s)
        a[tr.order] = a_s
        self.prev_a = a
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rec = {"call": self.calls, "mode": self.now, "lam": self.lam, "theta": self.theta, "cost": (m2l + dirn) / pos.shape[0], "time_s": (t1 - t0) if t0 else 0.0, "audit": None}
        if self.mode in ("geo_audit", "adaptive") and self.calls % self.audit_every == 0:
            err = (audit_fast if self.fast_audit else de.audit_e2e)(pos, a, self.audit_k, self.gen)
            torch.cuda.synchronize()
            rec["audit"], rec["audit_time_s"] = err, time.perf_counter() - t1
            self._control(err)
        self.calls += 1
        self.stats.append(rec)
        return a


def _exact_chunk_eager(src, tgt, eps: float = EPS):
    d = src[None] - tgt[:, None]
    dn = (d * d).sum(-1).sqrt().clamp(min=1e-9)
    w = (dn * (dn * dn + eps * eps) ** -1.5) / dn
    return (d * w[..., None]).sum(1)


_exact_state = {"fn": None, "failed": False}


def _exact_chunk(src, tgt):
    if not _exact_state["failed"]:
        try:
            if _exact_state["fn"] is None:
                _exact_state["fn"] = torch.compile(_exact_chunk_eager, dynamic=False)
            return _exact_state["fn"](src, tgt)
        except Exception:
            _exact_state["failed"] = True
    return _exact_chunk_eager(src, tgt)


@torch.no_grad()
def exact_fast(pos, idx, chunk=1024):
    src = pos.double()
    out = torch.empty(len(idx), 2, dtype=torch.float64, device=pos.device)
    for i in range(0, len(idx), chunk):
        t = src[idx[i:i + chunk]]
        if len(t) != chunk:
            t = torch.cat([t, t[-1:].expand(chunk - len(t), 2)])
        out[i:i + chunk] = _exact_chunk(src, t)[:len(idx[i:i + chunk])]
    return out


def audit_fast(pos, a_orig, K, gen):
    n = pos.shape[0]
    idx = torch.randperm(n, generator=gen, device=pos.device)[:K]
    a_ex = exact_fast(pos, idx, 128)
    return float((a_orig[idx].double() - a_ex).norm() / a_ex.norm())
