import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tiered.blend import TileLayout
from model.tiered.hierarchy import TieredField


def cic_density(pos, mass, mask, n, h, origin, moments=False):
    """Smooth (cloud-in-cell) mass density, (B,1,n,n); differentiable in pos. Mass falling off the grid is dropped.
    With moments=True returns (B,6,n,n): density plus CIC-weighted first and second offset moments about the cell centre
    (m dx/h, m dy/h, m dx^2/h^2, m dx dy/h^2, m dy^2/h^2, all per area); every channel is a continuous function of pos."""
    B, N = mask.shape
    s = (pos - origin) / h - 0.5
    i0 = torch.floor(s)
    f = s - i0
    i0 = i0.long()
    w = mass * mask
    C = 6 if moments else 1
    out = torch.zeros(B * n * n, C, device=pos.device, dtype=pos.dtype)
    bidx = torch.arange(B, device=pos.device)[:, None].expand(B, N)
    for dy in (0, 1):
        for dx in (0, 1):
            ix, iy = i0[..., 0] + dx, i0[..., 1] + dy
            ok = ((ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)).to(pos.dtype)
            wt = w * ok * (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
            flat = bidx * n * n + iy.clamp(0, n - 1) * n + ix.clamp(0, n - 1)
            if moments:
                ox, oy = s[..., 0] - (ix.to(s.dtype) + 0.0), s[..., 1] - (iy.to(s.dtype) + 0.0)
                val = torch.stack([wt, wt * ox, wt * oy, wt * ox * ox, wt * ox * oy, wt * oy * oy], -1)
            else:
                val = wt[..., None]
            out = out.index_add(0, flat.reshape(-1), val.reshape(-1, C))
    return out.view(B, n, n, C).permute(0, 3, 1, 2) / h ** 2


class TierNetE(nn.Module):
    """Upper tier as a scalar energy: padded-tile smooth density -> per-cell energy density (position-only, memoryless).
    Same attribute interface as TierNet where TieredField reads it."""

    hidden_ch = 0
    summary_ch = 0
    err = False
    pool = None
    energy = True

    def __init__(self, h, width=32, layers=6, out_scale=1.0, moments=False):
        super().__init__()
        self.h, self.out_scale, self.moments = h, out_scale, moments
        self.inp = nn.Conv2d(7 if moments else 2, width, 3, padding=1)
        self.mid = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(layers - 1))
        self.out = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rho):
        x = torch.cat([rho[:, :1], torch.log1p(rho[:, :1]), rho[:, 1:]], 1)
        h = F.gelu(self.inp(x))
        for c in self.mid:
            h = h + F.gelu(c(h))
        return self.out(h) * self.out_scale * self.h ** 2


class TieredFieldE(TieredField):
    """Each upper tier k contributes U_k = sum_cells e(density window) and the force on particle i is -dU_k/dx_i / m_i by
    autograd through the CIC deposit, so the upper-tier force is a gradient of one position-only scalar (no net work on closed
    paths; Newton-3 and energy bookkeeping up to the integrator). No mean removal."""

    def __init__(self, tier0, nets, n0, h0, G, halo, origin=None):
        nn.Module.__init__(self)
        self.tier0, self.nets, self.refiners = tier0, nn.ModuleList(nets), nn.ModuleList()
        self.n0, self.h0, self.G, self.periodic, self.mean_remove = n0, h0, G, False, False
        self.feedback = False
        if origin is None:
            origin = torch.full((2,), -n0 * h0 / 2)
        self.origin = torch.as_tensor(origin, dtype=torch.float32)
        self.layouts = [TileLayout(n0 >> k, G, halo, 0, h0 * 2 ** k, self.origin, False) for k in range(1, len(nets) + 1)]
        self.last_accs, self.last_errs = [], []

    def tier_energy(self, k, pos, mass, mask):
        net, L = self.nets[k - 1], self.layouts[k - 1]
        rho = cic_density(pos, mass, mask, L.n, L.h, self.origin.to(pos), self.nets[k - 1].moments)
        e = net(L.extract(rho).flatten(0, 2))
        e = e.reshape(rho.shape[0], L.T, L.T, 1, L.P, L.P)
        return L.interior(e).sum((1, 2, 3))

    def upper_force(self, pos, vel, mass, mask, state, occ=None):
        train = torch.is_grad_enabled()
        accs = []
        with torch.enable_grad():
            p = pos if pos.requires_grad else pos.detach().requires_grad_(True)
            for k in range(1, len(self.nets) + 1):
                U = self.tier_energy(k, p, mass, mask).sum()
                g = torch.autograd.grad(U, p, create_graph=train)[0]
                accs.append(-g / mass.clamp_min(1e-9)[..., None] * mask[..., None])
        self.last_accs = accs
        acc = sum(accs)
        if not train:
            acc = acc.detach()
        return acc, ([], [], [])
