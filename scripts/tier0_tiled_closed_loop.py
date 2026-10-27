"""Tier 0 trained inside the tiled hierarchy: differentiable 4x4 tiling of 64-unit windows (scripts/exp12_hier.S1Windows geometry, exact external
field = arm OR) with halo exchange of the recurrent field between neighbouring windows, plus the closed-loop rollout loss.

Recurrent state: the model's recurrent field is a per-window grid (G x G cells, cell width h). Windows are 2 tiles wide and spaced one tile apart,
so neighbouring grids are cell-aligned. exchange='halo': after every force call the new global field is assembled from the central tile
(tile_cells x tile_cells, the cells the window owns) of every window, and each window's recurrent input is the matching crop of that global field
(own tile + a rim copied from the adjacent owners; zeros outside the domain). One tile of rim per side, neighbours only: tileable, O(area).
exchange='zero': stateless variant, the recurrent input is zero every step. exchange='none': per-window state as in the old evaluation."""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from scripts import exp12_hier as h
from scripts import scatter_field as sf
from scripts import train_tiered as ttr

NT, NW, TILE, WIN, DT, EPS = h.NT, h.NW, h.TILE, h.WIN, ttr.DT, ttr.EPS


def exchange_field(f, tile_cells):
    """f (B*NW, C, G, G), windows indexed i*NT + j (i = x tile, j = y tile), dims (y, x). Returns the field every window sees after the halo exchange."""
    BW, C, G, _ = f.shape
    B, n = BW // NW, tile_cells
    r = (G - n) // 2
    core = f[:, :, r:r + n, r:r + n].reshape(B, NT, NT, C, n, n)
    g = core.permute(0, 3, 2, 4, 1, 5).reshape(B, C, NT * n, NT * n)
    g = F.pad(g, (r, r, r, r))
    w = g.unfold(2, G, n).unfold(3, G, n)
    return w.permute(0, 3, 2, 1, 4, 5).reshape(BW, C, G, G)


class TiledTier0:
    """Differentiable tier 0 on the tiling. force(pos, vel, mass, mask, fld) -> (acc, fld); fld is the per-window recurrent field (B*NW, C, G, G)."""

    def __init__(self, model, dev, chunk=32, exchange="halo", ckpt=True):
        assert exchange in ("halo", "zero", "none")
        self.m, self.t0, self.dev, self.chunk, self.exchange, self.ckpt = model, model.t0, dev, chunk, exchange, ckpt
        self.cen = h.centres(dev)
        self.n = int(round(TILE / model.h))
        assert (model.grid - self.n) % 2 == 0 and abs(model.extent - 2 * WIN) < 1e-6

    def init_state(self, B):
        f = self.m.init_field(B * NW, self.dev, torch.float32)
        return f[0] if isinstance(f, tuple) else f

    def _win(self, rel, v, mm, k, fld, g, H, Q, C):
        n = rel.shape[0]
        self.t0.set_external(self.m, phi0=torch.zeros(n, device=rel.device), g=g, H=H, high=(Q, C))
        self.t0.set_boundary(self.m, sdf=None, edge="open")
        return self.m.force(rel, v, mm, k, fld)

    def force(self, pos, vel, mass, mask, fld):
        B, N = mask.shape
        with torch.no_grad():
            g, H, Q, C = h.ext_for("OR", pos.detach(), vel, mass, mask, self.cen, None)
        rel, inw, _ = h.windows(pos, mask, self.cen)
        rel = rel.reshape(B * NW, N, 2)
        v = vel[:, None].expand(B, NW, N, 2).reshape(B * NW, N, 2)
        mm = mass[:, None].expand(B, NW, N).reshape(B * NW, N)
        k = inw.reshape(B * NW, N)
        out, nf = [], []
        use_ck = self.ckpt and torch.is_grad_enabled()
        for i in range(0, B * NW, self.chunk):
            sl = slice(i, i + self.chunk)
            args = (rel[sl], v[sl], mm[sl], k[sl], fld[sl], g[sl], H[sl], Q[sl], C[sl])
            a, f2 = cp.checkpoint(self._win, *args, use_reentrant=False) if use_ck else self._win(*args)
            out.append(a)
            nf.append(f2)
        a = torch.cat(out).reshape(B, NW, N, 2)
        acc = torch.gather(a, 1, h.owner(pos)[:, None, :, None].expand(B, 1, N, 2)).squeeze(1)
        f2 = torch.cat(nf)
        if self.exchange == "halo":
            f2 = exchange_field(f2, self.n)
        elif self.exchange == "zero":
            f2 = torch.zeros_like(f2)
        return acc * mask[..., None], f2


def step_weights(k, k_ref=20, decay=4.0):
    """Per-tick weights of the closed-loop loss: 1 up to the tick-20 scale, halving every `decay` ticks beyond it."""
    s = torch.arange(1, k + 1, dtype=torch.float32)
    return torch.where(s <= k_ref, torch.ones_like(s), 0.5 ** ((s - k_ref) / decay))


def verlet_step(tt, pos, vel, a_prev, fld, m, mk):
    pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
    a_new, fld = tt.force(pn, vel + DT * a_prev, m, mk, fld)
    return pn, vel + 0.5 * DT * (a_prev + a_new), a_new, fld


def rollout_terms(tt, d, idx, t0s, k, tb=0, grad=True):
    """Closed-loop rollout of k ticks from the truth state at t0s (per scene); returns per-tick mean squared position and velocity errors
    (lists of 0-dim tensors, differentiable when grad). tb > 0 truncates backprop (detach state) every tb ticks."""
    dev = d["P"].device
    tt_i = t0s[:, None] + torch.arange(k + 1, device=dev)[None]
    Pw, Vw = d["P"][idx[:, None], tt_i].float(), d["V"][idx[:, None], tt_i].float()
    m, mk = d["M"][idx].float(), d["mask"][idx].float()
    pos, vel = Pw[:, 0], Vw[:, 0]
    fld = tt.init_state(len(idx))
    a_prev, fld = tt.force(pos, vel, m, mk, fld)
    w = mk[:, :, None]
    denom = mk.sum() * 2
    lp, lv = [], []
    for s in range(k):
        pos, vel, a_prev, fld = verlet_step(tt, pos, vel, a_prev, fld, m, mk)
        lp.append((((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom)
        lv.append((((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom)
        if tb and (s + 1) % tb == 0:
            pos, vel, a_prev, fld = pos.detach(), vel.detach(), a_prev.detach(), fld.detach()
    return lp, lv


def closed_loop_loss(lp, lv, ref_p, ref_v, omega):
    """Each tick's errors are divided by the reference model's own mean squared error at that tick (so every tick counts as a relative error and
    the chaotic late ticks cannot dominate), then weighted by omega and averaged: equals ~1 for the reference model."""
    k = len(lp)
    t = torch.stack([lp[s] / ref_p[s] + 0.1 * lv[s] / ref_v[s] for s in range(k)])
    return (t * omega[:k].to(t.device)).sum() / (omega[:k].sum() * 1.1)
