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
import torch.nn as nn

from scripts import lj_sim
from scripts import scatter_field as sf

LS = 1.25
DT_M = 0.025
SUB = 5
GRID, EXTENT = 64, 32.0
CORE_MAX = 20.0
KNN = 24
SHIFTS = torch.tensor([[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)], dtype=torch.float64)


def build_model(ckpt=None, dev="cpu", energy=False, resid=False, vblind=False, norec=False, asym=False, oracle=False, nomesh=False, substeps=1):
    """Exp E ms_kp_pot_v_g128 built at G=128 (U-Net depth 5 so E checkpoints load), then run on a G=64, extent-32 grid (same cell h = 0.5)."""
    m = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=DT_M, momfix=True, in_scale=1.0,
                        potential=True, kernel=True, pp=2.0, split=True, verlet=True)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    m.grid, m.extent, m.h = GRID, EXTENT, EXTENT / GRID
    m.energy, m.vblind, m.norec, m.oracle, m.nomesh = energy, vblind, norec, oracle, nomesh
    m.dt = DT_M / substeps
    m.pair_sub = 1
    if energy:
        m.net = sf.UNet(1, 1, 32, 5)
        nn.init.zeros_(m.net.out.weight)
        nn.init.zeros_(m.net.out.bias)
    if resid:
        m.rnet = sf.UNet(3, 2, 32, 5)
        nn.init.zeros_(m.rnet.out.weight)
        nn.init.zeros_(m.rnet.out.bias)
    if asym:
        m.ppmlp_a = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(m.ppmlp_a[-1].weight)
        nn.init.zeros_(m.ppmlp_a[-1].bias)
    return m.to(dev)


def load_model(path, dev, nsub=None, pair_sub=1):
    """Rebuild a checkpoint's variant from its weights: energy-mode mesh (density-only 1-channel net) and optional residual net."""
    st = torch.load(path, map_location=dev, weights_only=False)["model"]
    a = torch.load(path, map_location="cpu", weights_only=False).get("args", {})
    m = build_model(None, dev, energy=st["net.inp.weight"].shape[1] == 1, resid="rnet.inp.weight" in st, vblind=a.get("vblind", False),
                    norec=a.get("norec", False), asym="ppmlp_a.0.weight" in st, oracle=a.get("oracle", False), nomesh=a.get("nomesh", False),
                    substeps=nsub or a.get("substeps", 1))
    m.pair_sub = pair_sub
    m.load_state_dict(st)
    return m


def layout(box):
    return tuple(max(1, math.ceil(float(b) / CORE_MAX)) for b in box)


class PeriodicTiles:
    def __init__(self, model, knn=KNN):
        self.m, self.knn = model, knn
        self.E = float(model.extent)
        self.pen = []

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

    def take_pen(self):
        """Mean (residual magnitude, residual curl, pair reciprocity, momentum) penalties over the force calls since the last take; zeros if absent."""
        out = torch.stack([torch.stack(p) for p in self.pen]).mean(0) if self.pen else torch.zeros(4)
        self.pen = []
        return out

    def pp_acc_asym(self, P, K):
        """Pair term with no built-in reciprocity: g depends on r and on both bodies' local densities (i, j order matters). Returns (acc, reciprocity penalty)."""
        m = self.m
        B, N = K.shape
        k = min(self.knn, N - 1)
        d_all = P[:, None, :, :] - P[:, :, None, :]
        r_all = torch.sqrt((d_all ** 2).sum(-1) + 1e-8)
        bad = (1 - K)[:, None, :] + torch.eye(N, device=P.device, dtype=P.dtype)[None]
        r_sel, idx = torch.topk(r_all + 1e6 * (bad > 0).to(P.dtype), k, dim=-1, largest=False)
        d = torch.gather(d_all, 2, idx[..., None].expand(B, N, k, 2))
        ok = (r_sel < 1e5).to(P.dtype) * K[:, :, None]
        r = r_sel.clamp(max=1e5)
        win = (1 - (r / m.pp).clamp(max=1.0) ** 2) ** 2
        rho = (win * ok).sum(2) / 8.0
        rho_j = torch.gather(rho[:, None, :].expand(B, N, N), 2, idx)
        rho_i = rho[:, :, None].expand(B, N, k)
        lr = torch.log(r + 0.05)
        g = m.ppmlp_a(torch.stack([r, lr, rho_i, rho_j], -1))[..., 0]
        g_sw = m.ppmlp_a(torch.stack([r, lr, rho_j, rho_i], -1))[..., 0]
        w = g * win * ok
        recip = (((g - g_sw) * win * ok) ** 2).sum() / ok.sum().clamp(min=1)
        return (w[..., None] * d / r[..., None]).sum(2), recip

    def pp_acc_lj(self, P, K):
        """Exact shifted-force LJ pair acceleration (model units) on the same top-knn neighbour set as the learned pair term."""
        B, N = K.shape
        k = min(self.knn, N - 1)
        d_all = P[:, None, :, :] - P[:, :, None, :]
        r_all = torch.sqrt((d_all ** 2).sum(-1) + 1e-8)
        bad = (1 - K)[:, None, :] + torch.eye(N, device=P.device, dtype=P.dtype)[None]
        r_sel, idx = torch.topk(r_all + 1e6 * (bad > 0).to(P.dtype), k, dim=-1, largest=False)
        d = torch.gather(d_all, 2, idx[..., None].expand(B, N, k, 2))
        ok = (r_sel < 1e5).to(P.dtype) * K[:, :, None] * (r_sel < self.m.pp).to(P.dtype)
        rl = r_sel.clamp(min=0.5, max=1e5) * LS
        f_rc = lj_sim._sf_consts(lj_sim.RC)[1]
        fmag = (24 * (2 * rl ** -13 - rl ** -7) - f_rc) / LS
        w = -fmag * ok
        return (w[..., None] * d / r_sel.clamp(min=1e-6)[..., None]).sum(2)

    def pair(self, P, K):
        if getattr(self.m, "oracle", False):
            return self.pp_acc_lj(P, K), None
        if hasattr(self.m, "ppmlp_a"):
            a, recip = self.pp_acc_asym(P, K)
            return a, recip
        return self.m.pp_acc(P, K, K, knn=self.knn), None

    def mesh_energy(self, rho):
        m = self.m
        G = m.grid
        rf = torch.fft.rfft2(rho, s=(2 * G, 2 * G))
        phi = torch.fft.irfft2(rf * m._kernel_fft(rho), s=(2 * G, 2 * G))[..., :G, :G] * m.h ** 2
        part = getattr(m, "mesh_part", "all")
        return (m.net(rho).sum() * m.h ** 2 if part != "kernel" else 0.0) + (0.5 * m.h ** 2 * (rho * phi).sum() if part != "net" else 0.0)

    def force_energy(self, pos, vel, mask, box, lay, field, parts=False):
        """Hole-1 fix (docs/theory/heterogeneous-tiered.md): mesh force = -d/dx of a scalar energy U = sum_cells psi(density) h^2 + 0.5 h^2 sum rho (K*rho),
        by autograd through CIC scatter, the net and the FFT kernel (position-only, memoryless), so it does no net work around a closed path.
        Plus the pair term, an optional soft-penalised residual force net (momentum channels, may depend on velocity), and global mean removal."""
        m = self.m
        S, N, _ = pos.shape
        pk = self.pack(pos, vel, mask, box, lay)
        P, V, K = pk["P"].to(field.dtype), pk["V"].to(field.dtype), pk["K"].to(field.dtype)
        train = torch.is_grad_enabled()
        which = getattr(self, "which", "all")
        x = None
        if which == "pair":
            g = torch.zeros_like(P)
        else:
            with torch.enable_grad():
                Pg = P if P.requires_grad else P.detach().requires_grad_(True)
                x = m.scatter(Pg, V, K, K)
                U = self.mesh_energy(x[:, :1])
                g = torch.autograd.grad(U, Pg, create_graph=train)[0]
        a_mesh = -g * K[..., None]
        if getattr(m, "nomesh", False):
            a_mesh = a_mesh * 0.0
        if which == "mesh":
            a_pp, recip = torch.zeros_like(P), None
        else:
            a_pp, recip = self.pair(P, K)
        a_pp = a_pp * K[..., None]
        a_res = torch.zeros_like(a_mesh)
        if hasattr(m, "rnet") and x is not None:
            r = m.rnet(x)
            a_res = m.gather(r, P) * K[..., None]
            curl = ((r[:, 1, 1:-1, 2:] - r[:, 1, 1:-1, :-2]) - (r[:, 0, 2:, 1:-1] - r[:, 0, :-2, 1:-1])) / 2
            self.pen.append([(a_res ** 2).sum() / K.sum(), (curl ** 2).mean(), a_res.new_zeros(()), a_res.new_zeros(())])
        a = self.unpack(a_mesh + a_pp + a_res, pk, S, N).to(pos.dtype)
        mk = mask[..., None]
        if recip is not None:
            z = a.new_zeros(())
            self.pen.append([z, z, recip, (((mk * a).sum(1) / mk.sum(1)) ** 2).sum(-1).mean()])
        a = (a - (mk * a).sum(1, keepdim=True) / mk.sum(1, keepdim=True)) * mk
        if parts:
            return a, field, self.unpack(a_pp, pk, S, N).to(pos.dtype), pk
        return a, field

    def force(self, pos, vel, mask, box, lay, field, parts=False):
        m = self.m
        if getattr(m, "energy", False):
            return self.force_energy(pos, vel, mask, box, lay, field, parts)
        S, N, _ = pos.shape
        pk = self.pack(pos, vel, mask, box, lay)
        P, V, K = pk["P"].to(field.dtype), pk["V"].to(field.dtype), pk["K"].to(field.dtype)
        x = m.scatter(P, V * (0.0 if getattr(m, 'vblind', False) else 1.0), K, K)
        ak = m.kernel_acc(x[:, :1])
        f = m.net(torch.cat([x, ak, field * (0.0 if getattr(m, 'norec', False) else 1.0)], 1))
        acc = m.neg_grad(f[:, :1]) + ak
        a_mesh = m.gather(acc, P)
        a_pp = m.pp_acc(P, K, K, knn=self.knn)
        a = self.unpack((a_mesh + a_pp) * K[..., None], pk, S, N).to(pos.dtype)
        mk = mask[..., None]
        a = (a - (mk * a).sum(1, keepdim=True) / mk.sum(1, keepdim=True)) * mk
        if parts:
            return a, f, self.unpack(a_pp * K[..., None], pk, S, N).to(pos.dtype), pk
        return a, f

    def step_respa(self, pos, vel, mask, box, lay, st):
        """Reversible multi-rate step: mesh kick at dt, pair term integrated with pair_sub velocity-Verlet substeps of dt/pair_sub."""
        field, a_mesh = st
        dt, n = self.m.dt, self.m.pair_sub
        self.which = "mesh"
        if a_mesh is None:
            a_mesh, field = self.force(pos, vel, mask, box, lay, field)
        v = vel + 0.5 * dt * a_mesh
        p = pos
        self.which = "pair"
        a_pp, _ = self.force(p, v, mask, box, lay, field)
        for _ in range(n):
            v = v + 0.5 * dt / n * a_pp
            p = (p + dt / n * v) % box[:, None, :]
            a_pp, _ = self.force(p, v, mask, box, lay, field)
            v = v + 0.5 * dt / n * a_pp
        self.which = "mesh"
        a_new, field = self.force(p, v, mask, box, lay, field)
        self.which = "all"
        return p, v + 0.5 * dt * a_new, (field, a_new)

    def step(self, pos, vel, mask, box, lay, st):
        if getattr(self.m, "pair_sub", 1) > 1:
            return self.step_respa(pos, vel, mask, box, lay, st)
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
    sub = model.dt / 0.005
    per = int(round(20 / sub))
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
            stats.append([j * sub] + true_stats(pl, vl, box_t, area))
            if log and (j // per) % 100 == 0:
                log(f"frame {j // per} T {stats[-1][1]:.3f} E/N {stats[-1][3]:.4f}")
        if j == frames * per:
            break
        p, v, st = pt.step(p, v, mask, bm, lay, st)
        if ramp is not None:
            target = ramp[0] + (ramp[1] - ramp[0]) * (j * sub) / total_steps
            temp = float(((v * LS) ** 2).sum()) / (2 * n)
            v = v * math.sqrt(1 + sub * 0.005 / tau * (target / temp - 1))
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
