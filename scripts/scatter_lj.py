"""Exp E scatter field on 2D periodic Lennard-Jones MD (docs/debugging/scatter-lj.md).

Mapping: LJ lengths are divided by LS = 1.25 so the shifted-force cutoff rc = 2.5 sigma is the model's pair radius pp = 2.0 (the model
is otherwise stock: U-Net on a grid of cell h = 0.5, FFT kernel, pp=2 pair term, potential head, Verlet). Time is unchanged, velocities and
accelerations divide by LS. The periodic box is cut into nx x ny cores of side <= CORE_MAX; every tile runs one copy of the model on a
G = 64 grid (extent 32, tile centre at the origin) and receives periodic ghost images of every body inside its extended square. Only own
bodies keep their force. Tiles are one batch (emulation of one device per tile); the recurrent field stays with its tile.
"""
import math

import numpy as np
import torch

from scripts import lj_sim
from scripts import scatter_field as sf

LS = 1.25
DT_M = 0.025
SUB = 5
GRID, EXTENT = 64, 32.0
CORE_MAX = 20.0
KNN = 24
SHIFTS = torch.tensor([[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)], dtype=torch.float64)


def build_model(ckpt=None, dev="cpu"):
    """Exp E ms_kp_pot_v_g128 built at G=128 (U-Net depth 5 so E checkpoints load), then run on a G=64, extent-32 grid (same cell h = 0.5)."""
    m = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=DT_M, momfix=True, in_scale=1.0,
                        potential=True, kernel=True, pp=2.0, split=True, verlet=True)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    m.grid, m.extent, m.h = GRID, EXTENT, EXTENT / GRID
    return m.to(dev)


def layout(box):
    return tuple(max(1, math.ceil(float(b) / CORE_MAX)) for b in box)


class PeriodicTiles:
    def __init__(self, model, knn=KNN):
        self.m, self.knn = model, knn
        self.E = float(model.extent)

    def init_field(self, S, lay, dev, dtype=torch.float32):
        return torch.zeros(S * lay[0] * lay[1], self.m.cf, GRID, GRID, device=dev, dtype=dtype)

    def pack(self, pos, vel, mask, box, lay):
        """pos/vel (S,N,2), mask (S,N), box (S,2), all systems share the tile layout. Returns padded tile tensors and the own-body map."""
        S, N, _ = pos.shape
        dev, dt = pos.device, pos.dtype
        n = torch.tensor(lay, device=dev)
        cs = box / n
        dmax = int(math.ceil(self.E / 2 / float(cs.min()) + 0.5)) - 1
        r = torch.arange(-dmax, dmax + 1, device=dev)
        offs = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).reshape(-1, 2)
        sh = SHIFTS.to(dev, dt)
        P = pos[:, :, None, :] + sh * box[:, None, None, :]
        it = torch.floor(P.detach() / cs[:, None, None, :]).long()
        tix = it[:, :, :, None, :] + offs
        valid = ((tix >= 0) & (tix < n)).all(-1)
        rel = P[:, :, :, None, :] - (tix + 0.5) * cs[:, None, None, None, :]
        inside = valid & (rel.detach().abs() < self.E / 2).all(-1) & (mask[:, :, None, None] > 0)
        e = inside.nonzero()
        s, b, si, di = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
        t2 = tix[s, b, si, di]
        T = S * lay[0] * lay[1]
        tid = s * (lay[0] * lay[1]) + t2[:, 0] * lay[1] + t2[:, 1]
        order = torch.argsort(tid, stable=True)
        s, b, si, di, tid = s[order], b[order], si[order], di[order], tid[order]
        counts = torch.bincount(tid, minlength=T)
        slot = torch.arange(len(tid), device=dev) - (torch.cumsum(counts, 0) - counts)[tid]
        nmax = max(2, int(counts.max()))
        flat = tid * nmax + slot
        Pt = pos.new_zeros(T * nmax, 2).index_put((flat,), rel[s, b, si, di])
        Vt = pos.new_zeros(T * nmax, 2).index_put((flat,), vel[s, b])
        Kt = pos.new_zeros(T * nmax).index_put((flat,), torch.ones(len(flat), device=dev, dtype=dt))
        own = (si == 4) & (di == (len(offs) - 1) // 2)
        return {"P": Pt.view(T, nmax, 2), "V": Vt.view(T, nmax, 2), "K": Kt.view(T, nmax), "flat": flat[own], "s": s[own], "b": b[own],
                "nmax": nmax, "ghosts": float(len(flat) - int(own.sum())) / T}

    def unpack(self, a, pk, S, N):
        out = a.new_zeros(S, N, 2)
        out[pk["s"], pk["b"]] = a.reshape(-1, 2)[pk["flat"]]
        return out

    def force(self, pos, vel, mask, box, lay, field, parts=False):
        m = self.m
        S, N, _ = pos.shape
        pk = self.pack(pos, vel, mask, box, lay)
        P, V, K = pk["P"].to(field.dtype), pk["V"].to(field.dtype), pk["K"].to(field.dtype)
        x = m.scatter(P, V, K, K)
        ak = m.kernel_acc(x[:, :1])
        f = m.net(torch.cat([x, ak, field], 1))
        acc = m.neg_grad(f[:, :1]) + ak
        a_mesh = m.gather(acc, P)
        a_pp = m.pp_acc(P, K, K, knn=self.knn)
        a = self.unpack((a_mesh + a_pp) * K[..., None], pk, S, N).to(pos.dtype)
        mk = mask[..., None]
        a = (a - (mk * a).sum(1, keepdim=True) / mk.sum(1, keepdim=True)) * mk
        if parts:
            return a, f, self.unpack(a_pp * K[..., None], pk, S, N).to(pos.dtype), pk
        return a, f

    def step(self, pos, vel, mask, box, lay, st):
        field, a_prev = st
        dt = self.m.dt
        if a_prev is None:
            a_prev, field = self.force(pos, vel, mask, box, lay, field)
        pn = (pos + dt * vel + 0.5 * dt ** 2 * a_prev) % box[:, None, :]
        a_new, field = self.force(pn, vel + dt * a_prev, mask, box, lay, field)
        dv = 0.5 * dt * (a_prev + a_new)
        return pn, vel + dv, (field, a_new)


def wrap_diff(a, b, box):
    d = a - b
    return d - box[..., None, :] * torch.round(d / box[..., None, :])


def true_stats(p, v, box_t, area):
    """(KE/N, PE/N, E/N, P) of a float64 state under the true shifted-force LJ (LJ units)."""
    n = p.shape[0]
    _, pe, vir = lj_sim.force(p, box_t)
    ke = 0.5 * (v ** 2).sum()
    return [float(ke) / n, float(pe) / n, (float(ke) + float(pe)) / n, (float(ke) + 0.5 * float(vir)) / area]


@torch.no_grad()
def rollout(model, pos0, vel0, box, frames, dev, ramp=None, total_steps=16000, tau=0.5, log=None):
    """Roll the model out from an LJ-unit state (float64 numpy). One recorded frame per SUB truth steps' worth of model steps
    (frame interval = 20 truth steps = 4 model steps). Berendsen driver toward `ramp` as in lj_sim.run, applied per model step.
    Returns pos, vel (frames+1, N, 2), stats (frames+1, 5) in LJ units, same layout as lj_sim.run, and diag dict."""
    pt = PeriodicTiles(model)
    n = pos0.shape[0]
    box_t = torch.tensor(box, device=dev, dtype=torch.float64)
    area = float(box[0] * box[1])
    lay = layout(box / LS)
    p = torch.tensor(pos0 / LS, device=dev, dtype=torch.float64)[None]
    v = torch.tensor(vel0 / LS, device=dev, dtype=torch.float64)[None]
    bm = (box_t / LS)[None]
    mask = torch.ones(1, n, device=dev, dtype=torch.float64)
    st = (pt.init_field(1, lay, dev), None)
    per = 20 // SUB
    fp, fv, stats, ratio = [], [], [], []
    ok = True
    for j in range(frames * per + 1):
        if j % per == 0:
            pl, vl = p[0] * LS, v[0] * LS
            fp.append(pl.cpu().numpy().astype(np.float32))
            fv.append(vl.cpu().numpy().astype(np.float32))
            if not (torch.isfinite(pl).all() and torch.isfinite(vl).all()):
                ok = False
                break
            stats.append([j * SUB] + true_stats(pl, vl, box_t, area))
            if log and (j // per) % 100 == 0:
                log(f"frame {j // per} T {stats[-1][1]:.3f} E/N {stats[-1][3]:.4f}")
        if j == frames * per:
            break
        p, v, st = pt.step(p, v, mask, bm, lay, st)
        if ramp is not None:
            target = ramp[0] + (ramp[1] - ramp[0]) * (j * SUB) / total_steps
            temp = float(((v * LS) ** 2).sum()) / (2 * n)
            v = v * math.sqrt(1 + SUB * 0.005 / tau * (target / temp - 1))
    return np.stack(fp), np.stack(fv), np.array(stats), ok


@torch.no_grad()
def two_body_force(model, d_lj, dev):
    """Full-model acceleration (LJ units, radial, positive = repulsive) of one of two bodies at separations d_lj (sigma), zero recurrent field,
    plus the pair-term-only curve (g(r) * window)."""
    pt = PeriodicTiles(model)
    S = len(d_lj)
    d = torch.tensor(d_lj / LS, device=dev, dtype=torch.float64)
    box = torch.full((S, 2), 30.0, device=dev, dtype=torch.float64)
    pos = torch.stack([torch.stack([15 - d / 2, torch.full_like(d, 15.0)], -1), torch.stack([15 + d / 2, torch.full_like(d, 15.0)], -1)], 1)
    mask = torch.ones(S, 2, device=dev, dtype=torch.float64)
    lay = layout(box[0])
    a, _ = pt.force(pos, torch.zeros_like(pos), mask, box, lay, pt.init_field(S, lay, dev))
    full = (-a[:, 0, 0] * LS).cpu().numpy()
    r = d.float()
    g = model.ppmlp(torch.stack([r, torch.log(r + 0.05)], -1))[..., 0]
    win = (1 - (r / model.pp).clamp(max=1.0) ** 2) ** 2
    return full, (-(g * win) * LS).detach().cpu().numpy()
