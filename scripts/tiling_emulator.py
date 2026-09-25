"""Tile emulator (one device) for the momentum-symmetric dual tree (scripts/dual_tree.py), 2D softened gravity.
Particles are partitioned into T tiles (x-strips or Morton-contiguous blocks, equal counts). Each tile runs its own dual-tree walk over
ordered node pairs (A, B) with A one of ITS nodes and B any node; it reads only its own particles plus what it records as imported:
node summaries (count, COM, quadrupole, size [, coarse-pass scale]) of nodes it does not fully own, and raw particles of foreign leaves that
enter a direct sum. The pair (B, A) is evaluated by B's tile from the same summaries; acceptance is the symmetric rule of dual_tree.py
(geometric theta, or the learned estimator with fixed lam), so momentum stays exact.
Designs (what a "node" is):
  A  per-tile roots: each tile builds a tree over its own particles with its own bounding box (node boundaries differ between tiles).
  B  shared grid, partial nodes: all tiles use the global root cell and level grid, node = (tile, cell) holding only that tile's particles.
  C  shared grid, merged nodes: nodes are the global tree's cells; a cell holding particles of several tiles has one merged summary that
     every tile sharing it evaluates for its own particles (its ordered pairs are computed by each sharing tile, restricted to own targets).
Communication is counted per tile as distinct imported nodes (fetched once per tile-step, cached) and distinct raw particles.
"""
import numpy as np
import torch

from scripts import adaptive_oracle as ao
from scripts import dual_tree as dt
from scripts import kernels

NODE_BYTES = 48
NODE_BYTES_EST = 56
PART_BYTES = 16


class TileTree:
    def __init__(self, pos, lmax, lo=None, D=None):
        dev = pos.device
        if lo is None:
            lo = pos.min(0).values
            D = float((pos.max(0).values - lo).max()) * (1 + 1e-6) + 1e-6
        n = 2 ** lmax
        ij = ((pos - lo) / D * n).long().clamp_(0, n - 1)
        key = ao.morton(ij[:, 0], ij[:, 1], lmax)
        order = torch.argsort(key)
        self.order = order
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


def global_grid(pos):
    lo = pos.min(0).values
    return lo, float((pos.max(0).values - lo).max()) * (1 + 1e-6) + 1e-6


def partition(pos, T, kind, lmax=18):
    N = len(pos)
    if kind == "strips":
        key = pos[:, 0]
    else:
        lo, D = global_grid(pos)
        n = 2 ** lmax
        ij = ((pos - lo) / D * n).long().clamp_(0, n - 1)
        key = ao.morton(ij[:, 0], ij[:, 1], lmax)
    order = torch.argsort(key)
    tid = torch.empty(N, dtype=torch.long, device=pos.device)
    tid[order] = torch.arange(N, device=pos.device) * T // N
    return tid


class Forest:
    """Node arrays of the whole emulated system (all tiles), in particle-sorted order. tid_sorted[p] = tile of sorted particle p."""

    def __init__(self, pos, tid, T, design, lmax=18):
        dev = pos.device
        self.T, self.design, self.lmax = T, design, lmax
        if design == "C":
            tr = ao.Tree(pos, lmax)
            fl = dt.Flat(tr)
            for k in ("count", "q", "start", "com", "level", "size", "child", "pnode"):
                setattr(self, k, getattr(fl, k))
            self.pos, self.order = tr.pos, tr.order
            self.roots = [0]
            self.node_tile = None
            self.tid_sorted = tid[tr.order]
        else:
            lo, D = global_grid(pos)
            cols = {k: [] for k in ("count", "q", "start", "com", "level", "size", "child", "pnode")}
            pos_l, order_l, roots, node_tile, tid_l = [], [], [], [], []
            noff, poff = 0, 0
            for s in range(T):
                idx = (tid == s).nonzero().squeeze(1)
                tr = TileTree(pos[idx], lmax, lo, D) if design == "B" else TileTree(pos[idx], lmax)
                fl = dt.Flat(tr)
                n = len(fl.count)
                cols["count"].append(fl.count)
                cols["q"].append(fl.q)
                cols["start"].append(fl.start + poff)
                cols["com"].append(fl.com)
                cols["level"].append(fl.level)
                cols["size"].append(fl.size)
                cols["child"].append(torch.where(fl.child >= 0, fl.child + noff, fl.child))
                cols["pnode"].append(fl.pnode + noff)
                pos_l.append(tr.pos)
                order_l.append(idx[tr.order])
                roots.append(noff)
                node_tile.append(torch.full((n,), s, dtype=torch.long, device=dev))
                tid_l.append(torch.full((len(idx),), s, dtype=torch.long, device=dev))
                noff += n
                poff += len(idx)
            for k, v in cols.items():
                setattr(self, k, torch.cat(v))
            self.pos, self.order = torch.cat(pos_l), torch.cat(order_l)
            self.roots, self.node_tile, self.tid_sorted = roots, torch.cat(node_tile), torch.cat(tid_l)
        self.n_nodes = len(self.count)
        self.lvl_nodes = [(self.level == l).nonzero().squeeze(1) for l in range(lmax)]
        self.N = len(self.pos)


def _direct(F, ga, gb, a, own_p, budget=30_000_000):
    if len(ga) == 0:
        return 0
    npair = F.count[ga] * F.count[gb]
    cs = torch.cumsum(npair, 0)
    i0, P, done = 0, len(ga), 0
    dev = ga.device
    while i0 < P:
        base = int(cs[i0 - 1]) if i0 else 0
        j0 = min(max(int(torch.searchsorted(cs, torch.tensor(base + budget, device=dev), right=True)), i0 + 1), P)
        npr = npair[i0:j0]
        E = int(npr.sum())
        rep = torch.arange(j0 - i0, device=dev).repeat_interleave(npr)
        k = torch.arange(E, device=dev) - (torch.cumsum(npr, 0) - npr)[rep]
        cbb = F.count[gb[i0:j0]][rep]
        pi = F.start[ga[i0:j0]][rep] + k // cbb
        pj = F.start[gb[i0:j0]][rep] + k % cbb
        keep = own_p[pi]
        pi, pj = pi[keep], pj[keep]
        a.index_add_(0, pi, kernels.current.pair(F.pos[pj] - F.pos[pi]))
        done += len(pi)
        i0 = j0
    return done


@torch.no_grad()
def tile_walk(F, s, theta, cap, accept_fn=None):
    """Forces on tile s's particles. Returns (a for sorted particles of tile s [indices p_own], p_own, stats)."""
    dev = F.pos.device
    own_p = F.tid_sorted == s
    if F.design == "C":
        cum = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(own_p.long(), 0)])
        cnt_own = cum[F.start + F.count] - cum[F.start]
    else:
        cnt_own = torch.where(F.node_tile == s, F.count, torch.zeros_like(F.count))
    n = F.n_nodes
    a_loc = torch.zeros(n, 2, dtype=torch.float64, device=dev)
    g_loc = torch.zeros(n, 4, dtype=torch.float64, device=dev)
    a_dir = torch.zeros(F.N, 2, dtype=torch.float64, device=dev)
    touched = torch.zeros(n, dtype=torch.bool, device=dev)
    dleaf = torch.zeros(n, dtype=torch.bool, device=dev)
    roots = torch.tensor(F.roots, dtype=torch.long, device=dev)
    ro = roots[cnt_own[roots] > 0]
    ga = ro.repeat_interleave(len(roots))
    gb = roots.repeat(len(ro))
    n_m2l = n_dir = 0
    while len(ga) > 0:
        ca, cb = F.count[ga], F.count[gb]
        touched[ga] = True
        touched[gb] = True
        r = F.com[gb] - F.com[ga]
        dist = r.norm(dim=1)
        sa, sb = F.size[ga], F.size[gb]
        acc = (torch.maximum(sa, sb) < theta * dist) & (ga != gb)
        if accept_fn is not None:
            acc = accept_fn(ga, gb, acc)
        if acc.any():
            r_, cb_, ga_ = r[acc], cb[acc].double(), ga[acc]
            g_, grad = kernels.current.g_grad(r_)
            a_loc.index_add_(0, ga_, g_ * cb_[:, None])
            g_loc.index_add_(0, ga_, grad * cb_[:, None])
            n_m2l += int(acc.sum())
        rest = ~acc
        leaf_a = (ca <= cap) | (F.level[ga] == F.lmax)
        leaf_b = (cb <= cap) | (F.level[gb] == F.lmax)
        dr = rest & leaf_a & leaf_b
        if dr.any():
            dleaf[gb[dr]] = True
            n_dir += _direct(F, ga[dr], gb[dr], a_dir, own_p)
        sp_a = rest & ~leaf_a & ((sa >= sb) | leaf_b)
        sp_b = rest & ~leaf_b & ((sb >= sa) | leaf_a)
        sp = sp_a | sp_b
        if not sp.any():
            break
        ga, gb, sp_a, sp_b = ga[sp], gb[sp], sp_a[sp], sp_b[sp]
        ea = torch.where(sp_a[:, None], F.child[ga], torch.stack([ga, *[torch.full_like(ga, -1)] * 3], 1))
        eb = torch.where(sp_b[:, None], F.child[gb], torch.stack([gb, *[torch.full_like(gb, -1)] * 3], 1))
        pa = ea.unsqueeze(2).expand(-1, 4, 4).reshape(-1)
        pb = eb.unsqueeze(1).expand(-1, 4, 4).reshape(-1)
        ok = (pa >= 0) & (pb >= 0)
        ga, gb = pa[ok], pb[ok]
        keep = cnt_own[ga] > 0
        ga, gb = ga[keep], gb[keep]
    for l in range(F.lmax):
        p = F.lvl_nodes[l]
        ch = F.child[p]
        v = ch >= 0
        pp = p[:, None].expand(-1, 4)[v]
        cc = ch[v]
        dcom = F.com[cc] - F.com[pp]
        gp = g_loc[pp]
        a_loc.index_add_(0, cc, a_loc[pp] - torch.stack([gp[:, 0] * dcom[:, 0] + gp[:, 1] * dcom[:, 1], gp[:, 2] * dcom[:, 0] + gp[:, 3] * dcom[:, 1]], 1))
        g_loc.index_add_(0, cc, gp)
    p_own = own_p.nonzero().squeeze(1)
    dx = F.pos[p_own] - F.com[F.pnode[p_own]]
    gl = g_loc[F.pnode[p_own]]
    a = a_dir[p_own] + a_loc[F.pnode[p_own]] - torch.stack([gl[:, 0] * dx[:, 0] + gl[:, 1] * dx[:, 1], gl[:, 2] * dx[:, 0] + gl[:, 3] * dx[:, 1]], 1)
    return a, p_own, {"m2l": n_m2l, "direct": n_dir, "touched": touched, "dleaf": dleaf, "cnt_own": cnt_own, "n_own_p": len(p_own)}


def import_counts(F, st):
    """(imported node summaries, imported raw particles) of a walk: touched nodes not fully owned; foreign particles of direct leaves."""
    cnt_own = st["cnt_own"]
    nodes = int((st["touched"] & (cnt_own < F.count)).sum())
    parts = int(((F.count - cnt_own)[st["dleaf"]]).sum())
    return nodes, parts


def run_tiled(F, theta, cap, accept_fn=None):
    """One force evaluation with every tile computing its own particles. Returns forces in sorted order and per-tile stats."""
    a = torch.zeros(F.N, 2, dtype=torch.float64, device=F.pos.device)
    tiles = []
    for s in range(F.T):
        a_s, p_own, st = tile_walk(F, s, theta, cap, accept_fn)
        a[p_own] = a_s
        tiles.append(st)
    return a, tiles


def summarize(F, tiles, est=False):
    rows = []
    for st in tiles:
        nodes, parts = import_counts(F, st)
        rows.append({"own": st["n_own_p"], "m2l": st["m2l"], "direct": st["direct"], "imp_nodes": nodes, "imp_parts": parts,
                     "imp_bytes": nodes * (NODE_BYTES_EST if est else NODE_BYTES) + parts * PART_BYTES})
    return rows


def halo_counts(pos, tid, T, radii):
    """Foreign particles within r of some own particle, upper bound via 3x3 dilation of occupied cells of size r (per tile)."""
    lo = pos.min(0).values
    out = {}
    for r in radii:
        ij = ((pos - lo) / r).long()
        M = int(ij.max()) + 3
        key = (ij[:, 0] + 1) * M + (ij[:, 1] + 1)
        per = []
        for s in range(T):
            own = tid == s
            ok = torch.unique(key[own])
            dil = torch.unique(torch.cat([ok + dx * M + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)]))
            per.append(int(torch.isin(key[~own], dil).sum()))
        out[r] = per
    return out


@torch.no_grad()
def cutoff_errors(pos, S, radii, chunk=256):
    """rel_l2 of the exact force truncated at r_c (pairs beyond r_c dropped) vs the full force, on targets S."""
    src = pos.double()
    full = torch.zeros(len(S), 2, dtype=torch.float64, device=pos.device)
    cut = {r: torch.zeros_like(full) for r in radii}
    for i in range(0, len(S), chunk):
        d = src[None] - src[S[i:i + chunk]][:, None]
        d2 = (d ** 2).sum(-1)
        f = d * ((d2 + ao.EPS ** 2) ** -1.5)[..., None]
        full[i:i + chunk] = f.sum(1)
        for r in radii:
            cut[r][i:i + chunk] = (f * (d2 < r * r)[..., None]).sum(1)
    return {r: float((cut[r] - full).norm() / full.norm()) for r in radii}
