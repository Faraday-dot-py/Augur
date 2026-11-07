import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tiered.blend import TileLayout, smoother
from model.tiered.moments import M, deposit_moments, moment_features


def core_mass(mom, layout):
    G, T = layout.G, layout.T
    return mom[:, M].reshape(mom.shape[0], T, G, T, G).sum((2, 4))


def active_tiles(mass_tiles, dilate=1, periodic=False):
    """A tile is active if its core holds mass or lies within `dilate` tiles of one that does."""
    a = (mass_tiles > 0).to(mass_tiles.dtype)[:, None]
    if dilate:
        k = 2 * dilate + 1
        a = F.pad(a, (dilate,) * 4, mode="circular" if periodic else "constant")
        a = F.max_pool2d(a, k, 1)
    return a[:, 0] > 0


def smooth_gate(rho, lo, hi):
    return smoother((rho - lo) / (hi - lo))[0]


class GatedRefine(nn.Module):
    """Residual refinement tier below tier 0 on cells of half size: potential phi_r from its own tile net,
    smooth gate g(rho) in [0,1] from local density; total potential g*phi_r so the force
    -(g grad phi_r + phi_r grad g) stays a gradient. Active only where the gate is nonzero (dilated by one tile)."""

    def __init__(self, net, n, G, halo, blend, h, origin, lo, hi, periodic=False, disp=True, cross=False, dilate=1):
        super().__init__()
        self.net, self.lo, self.hi, self.disp, self.cross, self.dilate = net, lo, hi, disp, cross, dilate
        self.layout = TileLayout(n, G, halo, blend, h, origin, periodic)

    def forward(self, pos, vel, mass, mask, occ=None):
        L = self.layout
        mom = deposit_moments(pos, vel, mass, mask, L.n, L.h, L.origin.to(pos), L.periodic, occ)
        gate = smooth_gate(mom[:, M:M + 1] / L.h ** 2, self.lo, self.hi)
        gt = L.extract(gate)
        act = active_tiles((gt[:, :, :, 0, L.halo:L.halo + L.G, L.halo:L.halo + L.G] > 0).any(-1).any(-1).to(pos.dtype), self.dilate, L.periodic)
        x = L.extract(moment_features(mom, L.h, self.disp, self.cross))
        B, T = x.shape[:2]
        coefs = x.new_zeros(B, T, T, self.net.ncoef, L.P, L.P)
        if act.any():
            coefs[act] = self.net(x[act])[0]
        phi, dphi, res = L.blend_eval(coefs, pos)
        g, dg, _ = L.blend_eval(gt, pos)
        return -(g[..., None] * dphi + phi[..., None] * dg) + g[..., None] * res, coefs, act
