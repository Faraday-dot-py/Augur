import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tiered.blend import TileLayout
from model.tiered.moments import M, deposit_moments, merge2x2, moment_features, nfeat
from model.tiered.sparse import active_tiles, core_mass


class ZeroTier0:
    """Tier-0 interface: dt, summary_ch, init_state(B, device, dtype) and force(pos, vel, mass, mask, state) -> (acc, state)."""

    dt = 0.1
    summary_ch = 0

    def init_state(self, B, device, dtype=torch.float32):
        return None

    def force(self, pos, vel, mass, mask, state):
        return torch.zeros_like(pos), state

    def summary(self):
        return None


class ScatterFieldTier0:
    """Black-box adapter around scripts.scatter_field.ScatterField (non-verlet); touches only its public force/init_field."""

    summary_ch = 0

    def __init__(self, model):
        self.model, self.dt = model, model.dt

    def init_state(self, B, device, dtype=torch.float32):
        f = self.model.init_field(B, device, dtype)
        return f[0] if isinstance(f, tuple) else f

    def force(self, pos, vel, mass, mask, state):
        return self.model.force(pos, vel, mass, mask, state)

    def summary(self):
        return None


def remove_mean(a, mass, mask):
    mm = (mass * mask)[..., None]
    return (a - (mm * a).sum(1, keepdim=True) / mm.sum(1, keepdim=True).clamp_min(1e-9)) * mask[..., None]


class TieredField(nn.Module):
    """Tier 0 (black box) plus upper tiers 1..K. Tier k cells are 2^k leaf cells wide; its GxG tile (plus halo
    copied from neighbouring parents) is the exact 2x2 moment merge of the tier k-1 grid. All tiers read the same
    state at t; their forces are summed and applied in one synchronous update. Each upper tier's force is the
    negative gradient of its blended polynomial potential (plus the soft residual channel) with the mass-weighted
    mean removed."""

    def __init__(self, tier0, nets, n0, h0, G, halo, blend, origin=None, periodic=False, disp=True, cross=False,
                 mean_remove=True, dilate=1, refiners=(), feedback=False, static=False, order3=False):
        super().__init__()
        self.tier0, self.nets, self.refiners = tier0, nn.ModuleList(nets), nn.ModuleList(refiners)
        self.n0, self.h0, self.G, self.periodic, self.disp, self.cross = n0, h0, G, periodic, disp, cross
        self.mean_remove, self.dilate, self.feedback, self.static, self.order3 = mean_remove, dilate, feedback, static, order3
        if origin is None:
            origin = torch.full((2,), -n0 * h0 / 2)
        self.origin = torch.as_tensor(origin, dtype=torch.float32)
        K = len(nets)
        assert n0 % (G * 2 ** K) == 0
        self.layouts, child = [], tier0.summary_ch
        for k, net in enumerate(nets, 1):
            n = n0 >> k
            self.layouts.append(TileLayout(n, G, halo, blend, h0 * 2 ** k, self.origin, periodic))
            ext = 3 if feedback and k < K else 0
            cin = nfeat(disp, cross, order3) + child + net.hidden_ch + ext
            assert net.inp.in_channels == cin, (k, net.inp.in_channels, cin)
            child = net.summary_ch

    @property
    def dt(self):
        return self.tier0.dt

    def init_state(self, B, device, dtype=torch.float32):
        hid = [torch.zeros(B, net.hidden_ch, L.n, L.n, device=device, dtype=dtype) for net, L in zip(self.nets, self.layouts)]
        return self.tier0.init_state(B, device, dtype), hid, [None] * len(self.nets)

    def moment_pyramid(self, pos, vel, mass, mask, occ=None):
        mom = [deposit_moments(pos, vel, mass, mask, self.n0, self.h0, self.origin.to(pos), self.periodic, occ)]
        for k in range(1, len(self.nets) + 1):
            mom.append(merge2x2(mom[-1], self.h0 * 2 ** (k - 1)))
        return mom

    def _ext(self, k, prev, B, device, dtype):
        L = self.layouts[k - 1]
        if not (self.feedback and k < len(self.nets)):
            return None
        if prev[k] is None:
            return torch.zeros(B, 3, L.n, L.n, device=device, dtype=dtype)
        v, g, _ = self.layouts[k].blend_eval(prev[k], L.centres(B, device, dtype))
        return torch.cat([v[..., None], g], -1).reshape(B, L.n, L.n, 3).permute(0, 3, 1, 2)

    def upper_force(self, pos, vel, mass, mask, state, occ=None, at=None):
        _, hid, prev = state
        B = pos.shape[0]
        if getattr(self, 'vblind_up', False):
            vel = torch.zeros_like(vel)
        mom = self.moment_pyramid(pos, vel, mass, mask, occ)
        acc, new_hid, coefs_all, act_all, accs, errs = 0, [], [], [], [], []
        child = self.tier0.summary()
        if child is not None:
            child = F.avg_pool2d(child, 2)
        for k, (net, L) in enumerate(zip(self.nets, self.layouts), 1):
            ch = [moment_features(mom[k], L.h, self.disp, self.cross, order3=self.order3)]
            cw = self.tier0.summary_ch if k == 1 else self.nets[k - 2].summary_ch
            if cw:
                ch.append(child if child is not None else ch[0].new_zeros(B, cw, L.n, L.n))
            if net.hidden_ch:
                ch.append(hid[k - 1])
            ext = self._ext(k, prev, B, pos.device, pos.dtype)
            if ext is not None:
                ch.append(ext)
            x = L.extract(torch.cat(ch, 1))
            T = L.T
            act = torch.ones(B, T, T, dtype=torch.bool, device=pos.device) if self.static else active_tiles(core_mass(mom[k], L), self.dilate, self.periodic)
            coefs = x.new_zeros(B, T, T, net.ncoef, L.P, L.P)
            summ = x.new_zeros(B, T, T, net.summary_ch, L.P, L.P)
            hnew = x.new_zeros(B, T, T, net.hidden_ch, L.P, L.P)
            if self.static:
                c, s_, hh = net(x.flatten(0, 2))
                coefs, summ, hnew = c.reshape_as(coefs), s_.reshape_as(summ), hh.reshape_as(hnew)
            elif act.any():
                coefs[act], summ[act], hnew[act] = net(x[act])
            phi, dphi, res = L.blend_eval(coefs, pos if at is None else at)
            a = -dphi + res
            a_t = remove_mean(a, mass, mask) if (self.mean_remove and at is None) else a
            acc = acc + a_t
            accs.append(a_t)
            if net.hidden_ch:
                am = act[:, None].repeat_interleave(L.G, 2).repeat_interleave(L.G, 3)
                new_hid.append(torch.where(am, L.interior(hnew), hid[k - 1]))
            else:
                new_hid.append(hid[k - 1])
            sg = L.interior(summ)
            errs.append(sg[:, -1:] if net.err else None)
            sg = sg * (mom[k][:, M:M + 1] > 0).to(pos.dtype)
            child = (net.pool(sg) if net.pool is not None else F.avg_pool2d(sg, 2)) if net.summary_ch else None
            coefs_all.append(coefs)
            act_all.append(act)
        self.last_accs, self.last_errs = accs, errs
        return acc, (new_hid, coefs_all, act_all)

    def slow_force(self, pos, vel, mass, mask, state, occ=None):
        acc, aux = self.upper_force(pos, vel, mass, mask, state, occ)
        for r in self.refiners:
            a, _, _ = r(pos, vel, mass, mask, occ)
            acc = acc + (remove_mean(a, mass, mask) if self.mean_remove else a)
        return acc * mask[..., None], aux

    def force(self, pos, vel, mass, mask, state, occ=None):
        a0, s0 = self.tier0.force(pos, vel, mass, mask, state[0])
        a1, (hid, coefs, act) = self.slow_force(pos, vel, mass, mask, state, occ)
        return (a0 + a1) * mask[..., None], (s0, hid, coefs), act

    def step(self, pos, vel, mass, mask, state, occ=None):
        a, state, _ = self.force(pos, vel, mass, mask, state, occ)
        dv = a * self.dt
        return pos + self.dt * (vel + 0.5 * dv), vel + dv, state, dv

    def step_verlet(self, pos, vel, mass, mask, state, a_prev=None, occ=None):
        dt = self.dt
        if a_prev is None:
            a_prev, state, _ = self.force(pos, vel, mass, mask, state, occ)
        pos_new = pos + dt * vel + 0.5 * dt ** 2 * a_prev
        a_new, state, _ = self.force(pos_new, vel + dt * a_prev, mass, mask, state, occ)
        return pos_new, vel + 0.5 * dt * (a_prev + a_new), state, a_new

    def step_respa(self, pos, vel, mass, mask, state, m, occ=None):
        """Slow (upper tier) half-kicks of m*dt around m fast tier-0 steps."""
        dt = self.dt
        a_slow, (hid, coefs, _) = self.slow_force(pos, vel, mass, mask, state, occ)
        vel = vel + 0.5 * m * dt * a_slow
        s0 = state[0]
        for _ in range(m):
            a, s0 = self.tier0.force(pos, vel, mass, mask, s0)
            dv = a * mask[..., None] * dt
            pos, vel = pos + dt * (vel + 0.5 * dv), vel + dv
        state = (s0, hid, coefs)
        a_slow, (hid, coefs, _) = self.slow_force(pos, vel, mass, mask, state, occ)
        return pos, vel + 0.5 * m * dt * a_slow, (s0, hid, coefs)
