"""Small residual corrector head on a frozen tier 0 (tiled hierarchy or plain window), plus the closed-loop rollout helpers and the
oracle (assimilation-style) force replacements used to bound what any corrector can win.

The head reads the frozen model's own rollout state: its predicted acceleration, local density (3x3 coarse cells), distance to the
boundary of the region whose window produced the force, step index since the rollout start and, optionally, the exact external-field
acceleration of the owner window. A tiny conv net on a coarse grid (O(N + G^2), translation equivariant) adds neighbourhood context.
Output = additive acceleration correction, last layer zero-initialised (identity at init), optionally made mean-free per scene
(same momentum treatment as the base's momfix)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import train_tiered as ttr

DT, EPS = ttr.DT, ttr.EPS


class ResidualCorrector(nn.Module):
    def __init__(self, width=32, ctx=8, grid=64, extent=128.0, use_ext=False, a_scale=0.1, out_scale=0.05, momfix=True, tick_scale=100.0):
        super().__init__()
        self.grid, self.extent, self.use_ext, self.a_scale, self.out_scale, self.momfix, self.tick_scale = grid, extent, use_ext, a_scale, out_scale, momfix, tick_scale
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=2, dilation=2), nn.GELU(),
                                  nn.Conv2d(16, ctx, 3, padding=1))
        d = 2 + 1 + 1 + 1 + (2 if use_ext else 0) + ctx
        self.mlp = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def cells(self, pos):
        i = torch.floor((pos + self.extent / 2) / (self.extent / self.grid)).long().clamp(0, self.grid - 1)
        return i[..., 1] * self.grid + i[..., 0]

    def forward(self, pos, a_f, mask, edge, tick, ext_g=None):
        B, N = mask.shape
        G = self.grid
        cell = self.cells(pos)
        flat = (torch.arange(B, device=pos.device)[:, None] * G * G + cell).reshape(-1)
        a_in = torch.asinh(a_f / self.a_scale)
        src = torch.cat([mask[..., None], a_in * mask[..., None]], -1).reshape(B * N, 3)
        acc = torch.zeros(B * G * G, 3, device=pos.device, dtype=pos.dtype).index_add(0, flat, src).reshape(B, G, G, 3).permute(0, 3, 1, 2)
        cnt = acc[:, :1]
        dens = F.avg_pool2d(cnt, 3, stride=1, padding=1) * 9
        grid_in = torch.cat([torch.log1p(cnt), acc[:, 1:] / cnt.clamp_min(1.0)], 1)
        ctx = self.conv(grid_in).permute(0, 2, 3, 1).reshape(B * G * G, -1)[flat].reshape(B, N, -1)
        dn = dens.reshape(B, G * G)[torch.arange(B, device=pos.device)[:, None], cell]
        f = [a_in, torch.log1p(dn)[..., None] / 3.0, (edge / 16.0).clamp(0.0, 2.0)[..., None], torch.full_like(edge, float(tick) / self.tick_scale)[..., None]]
        if self.use_ext:
            f.append(torch.asinh((ext_g if ext_g is not None else torch.zeros_like(a_f)) / self.a_scale))
        out = self.mlp(torch.cat(f + [ctx], -1)) * self.out_scale * mask[..., None]
        if self.momfix:
            out = (out - out.sum(1, keepdim=True) / mask.sum(1).clamp_min(1.0)[:, None, None]) * mask[..., None]
        return out


class Oracle:
    """Replaces the base force a_f by the exact force at the model's own state: kind 'lam' = a_f + value*(a_true - a_f) everywhere;
    'steps' = exact at ticks divisible by int(value); 'parts' = exact for a fixed random fraction `value` of the particles;
    'dense' = exact for the top fraction `value` of particles by neighbour count within 2 units."""

    def __init__(self, kind, value, seed=4738):
        self.kind, self.value, self.seed, self.sel = kind, value, seed, None

    def __call__(self, a_f, pos, m, mk, tick):
        at = ttr.true_acc(pos, m, mk).to(a_f.dtype)
        if self.kind == "lam":
            return a_f + self.value * (at - a_f)
        if self.kind == "steps":
            return at if tick % int(self.value) == 0 else a_f
        if self.kind == "parts":
            if self.sel is None or self.sel.shape != mk.shape:
                g = torch.Generator().manual_seed(self.seed)
                self.sel = (torch.rand(mk.shape, generator=g) < self.value).to(mk.device)
            sel = self.sel
        else:
            d2 = ((pos[:, :, None] - pos[:, None]) ** 2).sum(-1)
            near = ((d2 < 4.0) & (mk[:, None] > 0)).sum(-1).to(a_f.dtype) * mk - (1 - mk) * 1e9
            rank = near.argsort(1, descending=True).argsort(1)
            sel = rank < (self.value * mk.sum(1, keepdim=True)).ceil()
        return torch.where(sel[..., None], at, a_f)


def make_acc(base, corr=None, oracle=None):
    """acc(pos, vel, tick, m, mk, st) -> (a, st). The frozen base runs under no_grad (stop-gradient through it)."""
    def acc(pos, vel, tick, m, mk, st):
        with torch.no_grad():
            a_f, st, g = base.force(pos, vel, m, mk, st)
        a = a_f
        if oracle is not None:
            a = oracle(a_f, pos.detach(), m, mk, tick)
        if corr is not None:
            a = a + corr(pos, a_f, mk, base.edge(pos), tick, g)
        return a, st
    return acc


def verlet(acc, pos, vel, a_prev, st, tick, m, mk):
    pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
    a_new, st = acc(pn, vel + DT * a_prev, tick, m, mk, st)
    return pn, vel + 0.5 * DT * (a_prev + a_new), a_new, st


@torch.no_grad()
def rollout(acc, base, pos, vel, m, mk, steps, ticks):
    st = base.init_state(pos.shape[0])
    a_prev, st = acc(pos, vel, 0, m, mk, st)
    out = {0: (pos, vel)}
    for t in range(1, steps + 1):
        pos, vel, a_prev, st = verlet(acc, pos, vel, a_prev, st, t, m, mk)
        if t in ticks:
            out[t] = (pos, vel)
    return out


@torch.no_grad()
def score(acc, base, d, ticks=(10, 20, 50, 100), steps=None):
    """err@k (mean per-particle position error vs truth), dE@k = |E_model(k) - E_truth(0)| / |E_truth(0)| (the exp12_hier metric), per-scene means."""
    from scripts import scatter_field as sf

    T = d["P"].shape[1]
    steps = steps or max(ticks)
    ticks = tuple(k for k in ticks if k < T)
    m, mk = d["M"].float(), d["mask"].float()
    traj = rollout(acc, base, d["P"][:, 0].float(), d["V"][:, 0].float(), m, mk, max(ticks), ticks)
    cnt = d["mask"].sum(1)
    E0 = sf.energy(d["P"][:, 0], d["V"][:, 0], d["M"], d["mask"], EPS)
    res = {}
    for k in ticks:
        e = ((traj[k][0].double() - d["P"][:, k]).norm(dim=-1) * d["mask"]).sum(1) / cnt
        Em = sf.energy(traj[k][0].double(), traj[k][1].double(), d["M"], d["mask"], EPS)
        res[f"err@{k}"] = float(e.mean())
        res[f"dE@{k}"] = float(((Em - E0).abs() / E0.abs()).mean())
    return res


class TiledBase:
    """Frozen matched tier 0 on the 4x4 tiling of 64-unit windows (scripts/exp12_hier.S1Windows) with the exact external field (arm OR)."""

    def __init__(self, ckpt, dev, chunk=64):
        from scripts import exp12_hier as h

        self.h, self.dev = h, dev
        self.s1 = h.S1Windows(ckpt, dev, chunk=chunk)
        self.cen = h.centres(dev)

    def init_state(self, B):
        return self.s1.init_state(B)

    def force(self, pos, vel, m, mk, st):
        h = self.h
        ext = h.ext_for("OR", pos, vel, m, mk, self.cen, None)
        a, st = self.s1.force(pos, vel, m, mk, ext, st)
        B, N = mk.shape
        g = ext[0].reshape(B, h.NW, 2)
        g = torch.gather(g, 1, h.owner(pos)[..., None].expand(B, N, 2))
        return a, st, g

    def edge(self, pos):
        h = self.h
        rel = pos - self.cen[h.owner(pos)]
        return (h.TILE / 2 - rel.abs().amax(-1)).clamp_min(0.0)


class PlainBase:
    """Frozen single-window tier 0 (no tiling, no external field)."""

    def __init__(self, model):
        self.model = model

    def init_state(self, B):
        return self.model.init_field(B, next(self.model.parameters()).device, torch.float32)

    def force(self, pos, vel, m, mk, st):
        a, st = self.model.force(pos, vel, m, mk, st)
        return a, st, None

    def edge(self, pos):
        return (32.0 - pos.abs().amax(-1)).clamp(0.0, 32.0)
