import math

import torch
import torch.nn as nn

from scripts import scatter_field as sf

BITS = 19
OFF = 1 << (BITS - 1)
BSHIFT = 3 * BITS
DELTAS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
DKEY = torch.tensor([(dx << (2 * BITS)) + (dy << BITS) + dz for dx, dy, dz in DELTAS], dtype=torch.long)
CORNERS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
CKEY = torch.tensor([(a << (2 * BITS)) + (b << BITS) + c for a, b, c in CORNERS], dtype=torch.long)
SHARED_FROM = 3


def pack(bidx, u):
    return (bidx << BSHIFT) + (u[..., 0] << (2 * BITS)) + (u[..., 1] << BITS) + u[..., 2]


def unpack(key):
    m = (1 << BITS) - 1
    return torch.stack([(key >> (2 * BITS)) & m, (key >> BITS) & m, key & m], -1)


def neighbour_table(keys, query, dkey):
    """keys sorted unique (n,); query (q,); returns (q, len(dkey)) indices into keys, n where absent."""
    n = keys.shape[0]
    probe = query[:, None] + dkey.to(keys.device)[None]
    pos = torch.searchsorted(keys, probe.reshape(-1)).clamp(max=n - 1).view_as(probe)
    return torch.where(keys[pos] == probe, pos, torch.full_like(pos, n))


class SparseConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.lin = nn.Linear(27 * cin, cout)

    def forward(self, x, nb):
        xp = torch.cat([x, x.new_zeros(1, x.shape[1])], 0)
        return self.lin(xp[nb].flatten(1))


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.c2 = SparseConv(cin, cout), SparseConv(cout, cout)
        self.act = nn.GELU()

    def forward(self, x, nb):
        return self.act(self.c2(self.act(self.c1(x, nb)), nb))


class Level:
    pass


class SparseVoxelScatterField(nn.Module):
    """3D scatter field on occupied voxels plus a one-voxel halo. Bodies are deposited trilinearly (CIC) on voxel nodes of size h; a sparse U-Net (27-tap
    convolutions on the occupied set, children mean-pooled to parents, levels added until at most top_cap voxels remain) outputs a potential per node; acceleration
    is its central-difference gradient gathered trilinearly. At the coarsest level a learned radial kernel is summed directly over the (at most top_cap) voxel monopoles.
    Pair term: learned radial magnitude over the knn nearest bodies inside radius pp, found with a cell list (cell edge pp). Same Verlet/momfix step as sf.ScatterField."""

    def __init__(self, h=1.0, width=32, dt=0.1, pp=2.0, knn=16, top_cap=64, max_levels=16, cell_cap=32, momfix=True, eps=0.5):
        super().__init__()
        self.h, self.dt, self.pp, self.knn, self.top_cap, self.max_levels, self.cell_cap = h, dt, pp, knn, top_cap, max_levels, cell_cap
        self.momfix, self.eps = momfix, eps
        self.verlet, self.substeps, self.t0, self.opt = True, 1, None, set()
        self.cell_overflow = 0
        cin = 4 + 3
        self.inp = nn.Linear(cin, width)
        self.down = nn.ModuleList([Block(width, width) for _ in range(SHARED_FROM + 1)])
        self.up = nn.ModuleList([Block(2 * width, width) for _ in range(SHARED_FROM + 1)])
        self.lvl = nn.Linear(1, width)
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.kmlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.ppmlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        for m in (self.kmlp, self.ppmlp):
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    step = sf.ScatterField.step
    _step = sf.ScatterField._step

    def init_field(self, B, device, dtype=torch.float32):
        return (None, None)

    def structure(self, pos, mask):
        """Per-level sorted keys and neighbour/parent tables. Level 0 = unique CIC corners of the real bodies, dilated by one voxel."""
        B, N, _ = pos.shape
        dev = pos.device
        g = pos / self.h
        i0 = torch.floor(g)
        frac = g - i0
        u0 = i0.long() + OFF
        bidx = torch.arange(B, device=dev)[:, None].expand(B, N)
        real = mask > 0
        base = pack(bidx, u0)[real]
        ck = (base[:, None] + CKEY.to(dev)[None]).reshape(-1)
        occ = torch.unique(ck)
        s0 = torch.unique((occ[:, None] + DKEY.to(dev)[None]).reshape(-1))
        levels = []
        keys = s0
        for l in range(self.max_levels + 1):
            L = Level()
            L.keys = keys
            L.nb = neighbour_table(keys, keys, DKEY)
            levels.append(L)
            if int(torch.bincount(keys >> BSHIFT).max()) <= self.top_cap or l == self.max_levels:
                break
            b = keys >> BSHIFT
            u = unpack(keys & ((1 << BSHIFT) - 1)) >> 1
            pk = torch.unique(pack(b, u))
            L.parent = torch.searchsorted(pk, pack(b, u))
            L.nparent = pk.shape[0]
            keys = pk
        return levels, u0, frac, bidx, real

    def deposit(self, levels, u0, frac, bidx, real, vel, mass):
        mm = (mass * real)[..., None]
        val = torch.cat([mm, mm * vel], -1) / self.h ** 3
        keys = levels[0].keys
        x = val.new_zeros(keys.shape[0], 4)
        for (a, b, c), dk in zip(CORNERS, CKEY):
            w = (frac[..., 0] if a else 1 - frac[..., 0]) * (frac[..., 1] if b else 1 - frac[..., 1]) * (frac[..., 2] if c else 1 - frac[..., 2])
            idx = torch.searchsorted(keys, (pack(bidx, u0) + dk.to(keys.device))[real])
            x.index_add_(0, idx, (val[real] * w[real][:, None]))
        return x

    def node_pos(self, level, l):
        u = unpack(level.keys & ((1 << BSHIFT) - 1)).to(torch.float32)
        return ((u - (OFF >> l)) * 2 ** l) * self.h

    def coarse_kernel(self, level, l, mass_top):
        keys = level.keys
        b = keys >> BSHIFT
        x = self.node_pos(level, l)
        d = x[None, :, :] - x[:, None, :]
        r = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        ok = (b[:, None] == b[None, :]) & ~torch.eye(keys.shape[0], dtype=torch.bool, device=keys.device)
        g = self.kmlp(torch.stack([r, torch.log(r + self.h * 2 ** l)], -1))[..., 0]
        w = g * mass_top[None, :] * ok
        return (w[..., None] * d / r[..., None]).sum(1)

    def gather_corners(self, field, levels, u0, frac, bidx, real):
        keys = levels[0].keys
        out = field.new_zeros(*real.shape, field.shape[1])
        acc = 0
        for (a, b, c), dk in zip(CORNERS, CKEY):
            w = (frac[..., 0] if a else 1 - frac[..., 0]) * (frac[..., 1] if b else 1 - frac[..., 1]) * (frac[..., 2] if c else 1 - frac[..., 2])
            kk = pack(bidx, u0) + dk.to(keys.device)
            idx = torch.searchsorted(keys, kk.reshape(-1)).clamp(max=keys.shape[0] - 1).view_as(kk)
            acc = acc + field[idx] * w[..., None]
        return acc * real[..., None]

    def pair_acc(self, pos, mass, mask):
        B, N, _ = pos.shape
        dev = pos.device
        cell = torch.floor(pos / self.pp).long() + OFF
        bidx = torch.arange(B, device=dev)[:, None].expand(B, N)
        real = (mask > 0).reshape(-1)
        key = pack(bidx, cell).reshape(-1)
        key = torch.where(real, key, torch.full_like(key, (1 << 62)))
        order = torch.argsort(key, stable=True)
        sk = key[order]
        ukeys, counts = torch.unique_consecutive(sk, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        self.cell_overflow = int((counts[ukeys < (1 << 62)] > self.cell_cap).sum())
        pf, mm = pos.reshape(-1, 3), (mass * mask).reshape(-1)
        M, k = self.cell_cap, self.knn
        j = torch.arange(M, device=dev)
        out = torch.zeros_like(pf)
        qs = torch.nonzero(real).squeeze(1)
        Q = max(1, (1 << 24) // (27 * M))
        for s in range(0, qs.shape[0], Q):
            q = qs[s:s + Q]
            nk = key[q][:, None] + DKEY.to(dev)[None]
            ci = torch.searchsorted(ukeys, nk.reshape(-1)).clamp(max=ukeys.shape[0] - 1).view_as(nk)
            hit = ukeys[ci] == nk
            cnt = torch.where(hit, counts[ci], torch.zeros_like(ci)).clamp(max=M)
            idx = (starts[ci][..., None] + j).clamp(max=order.shape[0] - 1)
            valid = (j < cnt[..., None]).reshape(len(q), 27 * M)
            cand = order[idx].reshape(len(q), 27 * M)
            bad = ~valid | (cand == q[:, None])
            d = pf[cand] - pf[q][:, None]
            r_all = torch.sqrt((d ** 2).sum(-1) + 1e-8)
            kk = min(k, 27 * M)
            r_sel, i = torch.topk(r_all + 1e6 * bad.to(pf.dtype), kk, dim=-1, largest=False)
            ds = torch.gather(d, 1, i[..., None].expand(-1, -1, 3))
            mj = mm[torch.gather(cand, 1, i)]
            ok = (r_sel < self.pp).to(pf.dtype)
            r = r_sel.clamp(max=1e5)
            g = self.ppmlp(torch.stack([r, torch.log(r + 0.05)], -1))[..., 0]
            win = (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2
            w = g * win * mj * ok
            out[q] = (w[..., None] * ds / r[..., None]).sum(1)
        return out.view(B, N, 3)

    def force(self, pos, vel, mass, mask, field):
        levels, u0, frac, bidx, real = self.structure(pos, mask)
        x0 = self.deposit(levels, u0, frac, bidx, real, vel, mass)
        top = len(levels) - 1
        mass_l = [x0[:, 0] * self.h ** 3]
        for l in range(top):
            mass_l.append(torch.zeros(levels[l].nparent, device=pos.device, dtype=pos.dtype).index_add(0, levels[l].parent, mass_l[l]))
        ak = self.coarse_kernel(levels[top], top, mass_l[top])
        n0 = levels[0].keys.shape[0]
        mass0 = torch.zeros(n0, 1, device=pos.device, dtype=pos.dtype)
        mass0[:, 0] = x0[:, 0]
        h = self.inp(torch.cat([torch.log1p(x0[:, :1].clamp_min(0)), x0[:, 1:] / (1 + x0[:, :1].clamp_min(0)), x0[:, :1].new_zeros(n0, 3)], -1))
        feats = []
        for l, L in enumerate(levels):
            h = h + self.lvl(h.new_full((1, 1), float(l) / 8))
            h = self.down[min(l, SHARED_FROM)](h, L.nb)
            feats.append(h)
            if l < top:
                cnt = torch.zeros(L.nparent, 1, device=h.device, dtype=h.dtype).index_add(0, L.parent, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype))
                h = torch.zeros(L.nparent, h.shape[1], device=h.device, dtype=h.dtype).index_add(0, L.parent, h) / cnt
        akn = ak / (1 + ak.norm(dim=-1, keepdim=True))
        cur = feats[top] + self.inp(torch.cat([torch.zeros(feats[top].shape[0], 4, device=h.device, dtype=h.dtype), akn], -1))
        for l in range(top - 1, -1, -1):
            cur = self.up[min(l, SHARED_FROM)](torch.cat([feats[l], cur[levels[l].parent]], -1), levels[l].nb)
        phi = self.out(cur)
        nb = levels[0].nb
        pp_ = torch.cat([phi, phi.new_zeros(1, 1)], 0)
        idx = {d: i for i, d in enumerate(DELTAS)}
        grad = torch.cat([(pp_[nb[:, idx[(1, 0, 0)]]] - pp_[nb[:, idx[(-1, 0, 0)]]]),
                          (pp_[nb[:, idx[(0, 1, 0)]]] - pp_[nb[:, idx[(0, -1, 0)]]]),
                          (pp_[nb[:, idx[(0, 0, 1)]]] - pp_[nb[:, idx[(0, 0, -1)]]])], -1) / (2 * self.h)
        acc_nodes = -grad
        a = self.gather_corners(acc_nodes, levels, u0, frac, bidx, real)
        a = a + ak[self.top_parent(levels, u0, bidx, real)] * real[..., None]
        a = a + self.pair_acc(pos, mass, mask)
        if self.momfix:
            mm = (mass * mask)[..., None]
            a = a - (mm * a).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)
        return a * mask[..., None], field

    def top_parent(self, levels, u0, bidx, real):
        """Index of each body's coarsest-level voxel (the voxel holding its floor corner)."""
        keys0 = levels[0].keys
        idx = torch.searchsorted(keys0, pack(bidx, u0).reshape(-1)).clamp(max=keys0.shape[0] - 1)
        for L in levels[:-1]:
            idx = L.parent[idx]
        return idx.view_as(real)
