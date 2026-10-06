"""Opt-in bounce extension of ScatterField: uniform acceleration, wall term, separate contact term, optional substepped local terms.

All new parameters are zero-initialised (gscale=1), so an untrained BounceScatterField loaded from a plain ScatterField checkpoint reproduces it exactly.
Box: walls at box_lo / box_hi on both axes (centered coordinates), tokens are balls of unit mass.
"""
import torch
import torch.nn as nn

from scripts import scatter_field as sf


def _mlp():
    m = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    nn.init.zeros_(m[-1].weight)
    nn.init.zeros_(m[-1].bias)
    return m


OUT_SCALE = 10.0


def _win(r, rng):
    return (1 - (r / rng).clamp(max=1.0) ** 2) ** 2


class BounceScatterField(sf.ScatterField):
    def __init__(self, *args, sub=1, box_lo=-7.0, box_hi=7.0, pw=2.0, pc=2.0, **kw):
        super().__init__(*args, **kw)
        self.sub, self.box_lo, self.box_hi, self.pw, self.pc = sub, box_lo, box_hi, pw, pc
        self.gvec = nn.Parameter(torch.zeros(2))
        self.gscale = nn.Parameter(torch.ones(()))
        self.wmlp = _mlp()
        self.cmlp = _mlp()

    def wall_acc(self, pos):
        out = []
        for a in (0, 1):
            x = pos[..., a]
            lo, hi = x - self.box_lo, self.box_hi - x
            f = 0.0
            for sgn, d in ((1.0, lo), (-1.0, hi)):
                dc = d.clamp(min=-1.0)
                feat = torch.stack([dc, torch.log(dc.clamp(min=0.0) + 0.05)], -1)
                f = f + sgn * OUT_SCALE * self.wmlp(feat)[..., 0] * _win(dc.clamp(min=0.0), self.pw)
            out.append(f)
        return torch.stack(out, -1)

    def contact_acc(self, pos, mask):
        B, N = mask.shape
        d = pos[:, None, :, :] - pos[:, :, None, :]
        r = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        feat = torch.stack([r, torch.log(r + 0.05)], -1)
        g = OUT_SCALE * self.cmlp(feat)[..., 0] * _win(r, self.pc)
        ok = mask[:, None, :] * mask[:, :, None] * (1 - torch.eye(N, device=pos.device, dtype=pos.dtype))[None]
        return ((g * ok)[..., None] * d / r[..., None]).sum(2)

    def local_acc(self, pos, mask):
        a = OUT_SCALE * self.gvec + self.wall_acc(pos) + self.contact_acc(pos, mask)
        return a * mask[..., None]

    def base_force(self, pos, vel, mass, mask, field):
        a, field = super()._force(pos, vel, mass, mask, field)
        return a * self.gscale, field

    def _force(self, pos, vel, mass, mask, field):
        a, field = self.base_force(pos, vel, mass, mask, field)
        return a + self.local_acc(pos, mask), field

    def step(self, pos, vel, mass, mask, state):
        if self.sub == 1:
            return super().step(pos, vel, mass, mask, state)
        field, _ = state
        a_m, field = self.base_force(pos, vel, mass, mask, field)
        h = self.dt / self.sub
        p, v = pos, vel
        a_l = self.local_acc(p, mask)
        for _ in range(self.sub):
            v = v + 0.5 * h * (a_m + a_l)
            p = p + h * v
            a_l = self.local_acc(p, mask)
            v = v + 0.5 * h * (a_m + a_l)
        return p, v, (field, None), v - vel
