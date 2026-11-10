import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import scatter_field as sf

LEAF = 16
EDGE_IN = 6


def tree_order(pos, real, leaf=LEAF):
    """Balanced ball-tree order. pos (B,P,2), real (B,P) bool, P = leaf * 2**k. Returns perm (B,P): sorted slot -> input slot.
    Each group is halved by rank along its widest axis; padding sorts last, so every level pairs consecutive slots."""
    B, P = real.shape
    idx = torch.arange(P, device=pos.device)[None].expand(B, P)
    s = P
    while s > leaf:
        g = P // s
        gi = idx.reshape(B, g, s)
        c = torch.gather(pos[:, None].expand(B, g, P, 2), 2, gi[..., None].expand(B, g, s, 2))
        mk = torch.gather(real[:, None].expand(B, g, P), 2, gi)
        hi = torch.where(mk[..., None], c, c.new_tensor(-math.inf)).amax(2)
        lo = torch.where(mk[..., None], c, c.new_tensor(math.inf)).amin(2)
        axis = (hi - lo).argmax(-1)
        key = torch.gather(c, 3, axis[:, :, None, None].expand(B, g, s, 1))[..., 0]
        key = torch.where(mk, key, key.new_tensor(math.inf))
        order = torch.argsort(key, dim=-1, stable=True)
        idx = torch.gather(gi, 2, order).reshape(B, P)
        s //= 2
    return idx


class Block(nn.Module):
    """Ball attention with summed (un-normalised) weights: w_ij = softplus(q_i.k_j/sqrt(dh) + b(e_ij)) * M_j / ball, out_i = sum_j w_ij (v_j + ev_ij)."""

    def __init__(self, dim, heads, eh, ball):
        super().__init__()
        self.heads, self.dh, self.ball = heads, dim // heads, ball
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge1 = nn.Linear(EDGE_IN, eh)
        self.edge_b = nn.Linear(eh, heads)
        self.edge_v = nn.Linear(eh, dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, h, x, M):
        B, n, d = h.shape
        b, H, dh = self.ball, self.heads, self.dh
        G = n // b
        q, k, v = self.qkv(self.ln1(h)).view(B, G, b, 3, H, dh).unbind(3)
        xb, Mb = x.view(B, G, b, 2), M.view(B, G, b)
        rel = xb[:, :, None, :, :] - xb[:, :, :, None, :]
        r = torch.sqrt((rel ** 2).sum(-1) + 1e-12)
        lm = torch.log1p(Mb)
        e = torch.cat([rel / (r[..., None] + 0.25), torch.log(r + 0.05)[..., None], (r / 16.0)[..., None],
                       lm[:, :, None, :, None].expand(B, G, b, b, 1), lm[:, :, :, None, None].expand(B, G, b, b, 1)], -1)
        eh = F.gelu(self.edge1(e))
        ev = self.edge_v(eh).view(B, G, b, b, H, dh)
        logit = torch.einsum("bgihd,bgjhd->bgijh", q, k) / math.sqrt(dh) + self.edge_b(eh)
        w = F.softplus(logit) * (Mb / b)[:, :, None, :, None]
        msg = torch.einsum("bgijh,bgjhd->bgihd", w, v) + (w[..., None] * ev).sum(3)
        h = h + self.out(msg.reshape(B, n, d))
        return h + self.ffn(self.ln2(h))


class Pool(nn.Module):
    """Merge sibling pairs: node mass = sum, node position and features = mass-weighted means (features after a shared MLP of child feature, offset, log mass)."""

    def __init__(self, dim):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.me = nn.Linear(1, dim)

    def forward(self, h, x, M):
        B, n, d = h.shape
        hc, xc, Mc = h.view(B, n // 2, 2, d), x.view(B, n // 2, 2, 2), M.view(B, n // 2, 2)
        Mp = Mc.sum(2)
        den = Mp.clamp_min(1e-9)[..., None]
        xp = (xc * Mc[..., None]).sum(2) / den
        f = self.phi(torch.cat([hc, torch.asinh(xc - xp[:, :, None]), torch.log1p(Mc)[..., None]], -1))
        hp = (f * Mc[..., None]).sum(2) / den + self.me(torch.log1p(Mp)[..., None])
        return hp, xp, Mp


class Up(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim + 2, dim)

    def forward(self, hp, xp, hc, xc):
        par, xpp = hp.repeat_interleave(2, 1), xp.repeat_interleave(2, 1)
        return hc + self.lin(torch.cat([par, torch.asinh(xc - xpp)], -1))


class BallTreeAttention(nn.Module):
    """Grid-free force network: ball tree over the bodies, ball attention at every tree level (shared weights per pass), mass-weighted pooling up,
    refinement down, acceleration read per body. Same step interface as sf.ScatterField (Verlet, momfix), no pair term."""

    def __init__(self, dim=32, heads=4, eh=16, blocks=2, ball=LEAF, dt=0.1, momfix=True, extent=64.0):
        super().__init__()
        self.ball, self.dt, self.momfix, self.extent = ball, dt, momfix, extent
        self.verlet, self.substeps, self.t0, self.opt = True, 1, None, set()
        self.embed = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.down = nn.ModuleList([Block(dim, heads, eh, ball) for _ in range(blocks)])
        self.pool = Pool(dim)
        self.up = Up(dim)
        self.upblk = nn.ModuleList([Block(dim, heads, eh, ball) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    step = sf.ScatterField.step
    _step = sf.ScatterField._step

    def init_field(self, B, device, dtype=torch.float32):
        return (None, None)

    def force(self, pos, vel, mass, mask, field):
        B, N = mask.shape
        dev = pos.device
        cnt = mask.sum(1)
        n_eff = max(int(cnt.max()), 1)
        k = max(0, math.ceil(math.log2(n_eff / self.ball)))
        P = self.ball * 2 ** k
        if bool((cnt == N).all()):
            take = torch.arange(N, device=dev)[None].expand(B, N)
        else:
            take = torch.argsort((mask <= 0).to(torch.uint8), dim=1, stable=True)
        take = take[:, :P]
        if take.shape[1] < P:
            take = torch.cat([take, torch.full((B, P - take.shape[1]), N, device=dev, dtype=take.dtype)], 1)
        ext = lambda t: torch.cat([t, t.new_zeros(B, 1, *t.shape[2:])], 1)
        g = lambda t: torch.gather(ext(t), 1, take.reshape(B, P, *([1] * (t.dim() - 2))).expand(B, P, *t.shape[2:]))
        pos_p, vel_p, m_p = g(pos), g(vel), g(mass * mask)
        real = (take < N) & (m_p > 0)
        perm = tree_order(pos_p, real, self.ball)
        s = lambda t: torch.gather(t, 1, perm.reshape(B, P, *([1] * (t.dim() - 2))).expand(B, P, *t.shape[2:]))
        x, v, M = s(pos_p), s(vel_p), s(m_p) * s(real.to(pos.dtype))
        h = self.embed(torch.cat([v, torch.log1p(M)[..., None]], -1))
        hs, xs, Ms = [], [], []
        for lvl in range(k + 1):
            for blk in self.down:
                h = blk(h, x, M)
            hs.append(h)
            xs.append(x)
            Ms.append(M)
            if lvl < k:
                h, x, M = self.pool(h, x, M)
        cur = hs[k]
        for lvl in range(k - 1, -1, -1):
            cur = self.up(cur, xs[lvl + 1], hs[lvl], xs[lvl])
            for blk in self.upblk:
                cur = blk(cur, xs[lvl], Ms[lvl])
        a_sorted = self.head(cur)
        slot = torch.gather(take, 1, perm)
        a_ext = a_sorted.new_zeros(B, N + 1, 2).scatter(1, slot[..., None].expand(B, P, 2), a_sorted)
        a = a_ext[:, :N]
        if self.momfix:
            mm = (mass * mask)[..., None]
            a = a - (mm * a).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)
        return a * mask[..., None], field
