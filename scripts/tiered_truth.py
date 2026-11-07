"""Truth physics for the heterogeneous tiered architecture (docs/theory/heterogeneous-tiered.md).
Batched fp64 torch (GPU), centred coordinates, shapes (B, N, 2) with a 0/1 `mask`. Softened gravity (G=1, eps=0.5, same law as
scripts/scatter_scenes.nbody_truth), shifted-force LJ (as scripts/lj_sim.py), penalty contact (as scatter_scenes.bounce_forces),
bonded bodies, penalty walls, periodic minimum image, a smooth external field (ExtField: value/Jacobian/Hessian of the
acceleration), direct-sum far-field expansion about a window centre, and a cell-multipole gravity for larger N."""
import math

import torch

EPS = 0.5
LJ_RC = 2.5
PAIR_BUDGET = 2 ** 25


def _sf_consts(rc):
    return 4 * (rc ** -12 - rc ** -6), 24 * (2 * rc ** -13 - rc ** -7)


def _rows(B, N, budget=PAIR_BUDGET):
    return max(1, budget // max(1, B * N))


def _pair_iter(pos, mask, period=None, budget=PAIR_BUDGET):
    """Yields (i0, i1, d, r2, ok): d[b,q,j] = pos_j - pos_i (minimum image if period), ok excludes self and masked pairs."""
    B, N = mask.shape
    Q = _rows(B, N, budget)
    j = torch.arange(N, device=pos.device)
    for i0 in range(0, N, Q):
        i1 = min(N, i0 + Q)
        d = pos[:, None, :, :] - pos[:, i0:i1, None, :]
        if period is not None:
            p = period[:, None, None, :]
            d = d - p * torch.round(d / p)
        r2 = (d ** 2).sum(-1)
        ok = (mask[:, i0:i1, None] * mask[:, None, :]) > 0
        ok = ok & (j[None, None, :] != torch.arange(i0, i1, device=pos.device)[None, :, None])
        yield i0, i1, d, r2, ok


def grav_acc(pos, mass, mask, eps=EPS, g=1.0, period=None):
    B, N = mask.shape
    out = torch.zeros_like(pos)
    mm = mass * mask
    for i0, i1, d, r2, ok in _pair_iter(pos, mask, period):
        w = torch.where(ok, (r2 + eps ** 2) ** -1.5, torch.zeros_like(r2)) * mm[:, None, :]
        out[:, i0:i1] = g * (d * w[..., None]).sum(2)
    return out * mask[..., None]


def grav_pe(pos, mass, mask, eps=EPS, g=1.0, period=None):
    pe = pos.new_zeros(mask.shape[0])
    mm = mass * mask
    for i0, i1, d, r2, ok in _pair_iter(pos, mask, period):
        u = torch.where(ok, torch.rsqrt(r2 + eps ** 2), torch.zeros_like(r2)) * mm[:, i0:i1, None] * mm[:, None, :]
        pe = pe - 0.5 * g * u.sum((1, 2))
    return pe


def lj_acc(pos, mass, mask, rc=LJ_RC, period=None):
    u_rc, f_rc = _sf_consts(rc)
    out = torch.zeros_like(pos)
    for i0, i1, d, r2, ok in _pair_iter(pos, mask, period):
        ok = ok & (r2 < rc * rc) & (r2 > 1e-12)
        r2s = torch.where(ok, r2, torch.ones_like(r2))
        r = r2s.sqrt()
        ir6 = r2s ** -3
        fm = torch.where(ok, 24 * (2 * ir6 * ir6 - ir6) / r - f_rc, torch.zeros_like(r))
        out[:, i0:i1] = -((fm / r)[..., None] * d).sum(2)
    return out / mass.clamp_min(1e-9)[..., None] * mask[..., None]


def lj_pe(pos, mask, rc=LJ_RC, period=None):
    u_rc, f_rc = _sf_consts(rc)
    pe = pos.new_zeros(mask.shape[0])
    for i0, i1, d, r2, ok in _pair_iter(pos, mask, period):
        ok = ok & (r2 < rc * rc) & (r2 > 1e-12)
        r2s = torch.where(ok, r2, torch.ones_like(r2))
        r = r2s.sqrt()
        ir6 = r2s ** -3
        u = torch.where(ok, 4 * (ir6 * ir6 - ir6) - u_rc + f_rc * (r - rc), torch.zeros_like(r))
        pe = pe + 0.5 * u.sum((1, 2))
    return pe


def contact_acc(pos, mass, mask, radius, k, kmat=None, period=None):
    """Quadratic penalty between equal spheres (force k*pen^2, pen = 2*radius - r), k_ij = mean of per-particle stiffness."""
    out = torch.zeros_like(pos)
    km = torch.ones_like(mask) if kmat is None else kmat
    for i0, i1, d, r2, ok in _pair_iter(pos, mask, period):
        ok = ok & (r2 < (2 * radius) ** 2) & (r2 > 1e-12)
        r = torch.where(ok, r2, torch.ones_like(r2)).sqrt()
        pen = torch.where(ok, 2 * radius - r, torch.zeros_like(r))
        kij = k * 0.5 * (km[:, i0:i1, None] + km[:, None, :])
        out[:, i0:i1] = -((kij * pen ** 2 / r)[..., None] * d).sum(2)
    return out / mass.clamp_min(1e-9)[..., None] * mask[..., None]


def contact_pe(pos, mask, radius, k, kmat=None, period=None):
    pe = pos.new_zeros(mask.shape[0])
    km = torch.ones_like(mask) if kmat is None else kmat
    for i0, i1, d, r2, ok in _pair_iter(pos, mask, period):
        ok = ok & (r2 < (2 * radius) ** 2) & (r2 > 1e-12)
        r = torch.where(ok, r2, torch.ones_like(r2)).sqrt()
        pen = torch.where(ok, 2 * radius - r, torch.zeros_like(r))
        pe = pe + 0.5 * (k * 0.5 * (km[:, i0:i1, None] + km[:, None, :]) * pen ** 3 / 3).sum((1, 2))
    return pe


def wall_acc(pos, mass, mask, lo, hi, radius, k):
    lo, hi = lo[:, None, :], hi[:, None, :]
    pl = (lo + radius - pos).clamp_min(0)
    ph = (pos - (hi - radius)).clamp_min(0)
    return k * (pl ** 2 - ph ** 2) / mass.clamp_min(1e-9)[..., None] * mask[..., None]


def wall_pe(pos, mask, lo, hi, radius, k):
    lo, hi = lo[:, None, :], hi[:, None, :]
    pl = (lo + radius - pos).clamp_min(0)
    ph = (pos - (hi - radius)).clamp_min(0)
    return (k / 3 * (pl ** 3 + ph ** 3)).sum(-1).mul(mask).sum(-1)


def bond_terms(pos, vel, mass, mask, idx, k, rest, damp, bmask, brk=None):
    """Springs with optional damping along the bond; with brk (stretch ratio) a bond carries no force beyond brk*rest (state-free
    fracture: the spring potential is capped there). Returns acc (B,N,2) and spring PE (B,)."""
    B, N, _ = pos.shape
    i, j = idx[..., 0], idx[..., 1]
    gi = lambda a, ix: torch.gather(a, 1, ix[..., None].expand(-1, -1, 2))
    d = gi(pos, j) - gi(pos, i)
    l = d.norm(dim=-1).clamp_min(1e-9)
    n = d / l[..., None]
    rv = ((gi(vel, j) - gi(vel, i)) * n).sum(-1)
    live = bmask if brk is None else bmask * (l <= brk * rest).to(l.dtype)
    f = (k * (l - rest) + damp * rv) * live
    fv = f[..., None] * n
    out = torch.zeros_like(pos)
    out = out.scatter_add(1, i[..., None].expand(-1, -1, 2), fv).scatter_add(1, j[..., None].expand(-1, -1, 2), -fv)
    lc = l if brk is None else torch.minimum(l, brk * rest)
    pe = (0.5 * k * (lc - rest) ** 2 * bmask).sum(1)
    return out / mass.clamp_min(1e-9)[..., None] * mask[..., None], pe


class Physics:
    """Near-field law set. Any subset of: grav, lj, contact, bonds, walls, g_vec, drag; `period` (B,2) makes pairwise laws
    use the minimum image (wrap-around halo). `kmat` (B,N) is a per-particle stiffness (material channel)."""

    def __init__(self, grav=None, lj=None, contact=None, bonds=None, walls=None, period=None, g_vec=None, drag=0.0, kmat=None):
        self.grav, self.lj, self.contact, self.bonds, self.walls = grav, lj, contact, bonds, walls
        self.period, self.g_vec, self.drag, self.kmat = period, g_vec, drag, kmat

    def acc(self, pos, vel, mass, mask):
        a = torch.zeros_like(pos)
        if self.grav:
            a = a + grav_acc(pos, mass, mask, self.grav.get("eps", EPS), self.grav.get("g", 1.0), self.period)
        if self.lj:
            a = a + lj_acc(pos, mass, mask, self.lj.get("rc", LJ_RC), self.period)
        if self.contact:
            a = a + contact_acc(pos, mass, mask, self.contact["radius"], self.contact["k"], self.kmat, self.period)
        if self.bonds:
            a = a + bond_terms(pos, vel, mass, mask, **self.bonds)[0]
        if self.walls:
            a = a + wall_acc(pos, mass, mask, **self.walls)
        if self.g_vec is not None:
            a = a + torch.as_tensor(self.g_vec, device=pos.device, dtype=pos.dtype) * mask[..., None]
        if self.drag:
            a = a - self.drag * vel * mask[..., None]
        return a

    def pe(self, pos, vel, mass, mask):
        e = pos.new_zeros(mask.shape[0])
        if self.grav:
            e = e + grav_pe(pos, mass, mask, self.grav.get("eps", EPS), self.grav.get("g", 1.0), self.period)
        if self.lj:
            e = e + lj_pe(pos, mask, self.lj.get("rc", LJ_RC), self.period)
        if self.contact:
            e = e + contact_pe(pos, mask, self.contact["radius"], self.contact["k"], self.kmat, self.period)
        if self.bonds:
            e = e + bond_terms(pos, vel, mass, mask, **self.bonds)[1]
        if self.walls:
            e = e + wall_pe(pos, mask, **self.walls)
        if self.g_vec is not None:
            gv = torch.as_tensor(self.g_vec, device=pos.device, dtype=pos.dtype)
            e = e - (mass * mask * (pos * gv).sum(-1)).sum(-1)
        return e

    def energy(self, pos, vel, mass, mask):
        ke = 0.5 * (mass * mask * (vel ** 2).sum(-1)).sum(-1)
        return ke + self.pe(pos, vel, mass, mask)

    @property
    def conservative(self):
        return not self.drag and not (self.bonds and bool((self.bonds["damp"] != 0).any()))


class ExtField:
    """Smooth external acceleration a(x,t) = (1 + rate*t) * (a0 + J x + 0.5 Q[x,x]) with x relative to the window centre.
    Potential-derived (curl-free) when J symmetric and Q fully symmetric (always true for far-field expansions of gravity)."""

    def __init__(self, a0, J, Q, rate=None):
        self.a0, self.J, self.Q = a0, J, Q
        self.rate = torch.zeros(a0.shape[0], device=a0.device, dtype=a0.dtype) if rate is None else rate

    @staticmethod
    def zero(B, device, dtype=torch.float64):
        z = lambda *s: torch.zeros(B, *s, device=device, dtype=dtype)
        return ExtField(z(2), z(2, 2), z(2, 2, 2))

    @staticmethod
    def random(B, gen, device, a_scale=0.3, j_scale=0.02, q_scale=0.0, rate_scale=0.0, curl=0.0, dtype=torch.float64):
        r = lambda *s: torch.randn(B, *s, generator=gen, device=device, dtype=dtype)
        J = r(2, 2) * j_scale
        J = 0.5 * (J + J.transpose(1, 2)) + curl * 0.5 * (J - J.transpose(1, 2))
        Q = r(2, 2, 2)
        S = sum(Q.permute(0, *p) for p in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))) / 6 * q_scale
        return ExtField(r(2) * a_scale, J, S, r() * rate_scale)

    def acc(self, pos, t=0.0):
        s = (1 + self.rate * t)[:, None, None]
        a = self.a0[:, None, :] + torch.einsum("bij,bnj->bni", self.J, pos) + 0.5 * torch.einsum("bijk,bnj,bnk->bni", self.Q, pos, pos)
        return a * s

    def pe(self, pos, mass, mask, t=0.0):
        s = (1 + self.rate * t)[:, None]
        phi = -(torch.einsum("bi,bni->bn", self.a0, pos) + 0.5 * torch.einsum("bij,bni,bnj->bn", self.J, pos, pos)
                + torch.einsum("bijk,bni,bnj,bnk->bn", self.Q, pos, pos, pos) / 6)
        return (mass * mask * phi * s).sum(-1)

    def channels(self, G, extent, t=0.0, dtype=torch.float32):
        """(B, 6, G, G): acceleration (2) and its Jacobian (4) at grid cell centres, the tier-0 external-field input channels."""
        h = extent / G
        c = (torch.arange(G, device=self.a0.device, dtype=self.a0.dtype) + 0.5) * h - extent / 2
        X = torch.stack(torch.meshgrid(c, c, indexing="xy"), -1).reshape(1, G * G, 2).expand(self.a0.shape[0], -1, -1)
        s = (1 + self.rate * t)[:, None, None]
        a = self.acc(X, t)
        jac = (self.J[:, None] + torch.einsum("bijk,bnk->bnij", self.Q, X)) * s[..., None]
        ch = torch.cat([a, jac.reshape(a.shape[0], G * G, 4)], -1)
        return ch.permute(0, 2, 1).reshape(a.shape[0], 6, G, G).to(dtype)

    def index(self, sel):
        return ExtField(self.a0[sel], self.J[sel], self.Q[sel], self.rate[sel])


def _fkl(d, eps):
    """Derivatives wrt d of f(d) = d (|d|^2+eps^2)^-1.5: f (…,2), df[i,k] (…,2,2), ddf[i,k,l] (…,2,2,2)."""
    s = (d ** 2).sum(-1) + eps ** 2
    I = torch.eye(2, device=d.device, dtype=d.dtype)
    s15, s25, s35 = s ** -1.5, s ** -2.5, s ** -3.5
    f = d * s15[..., None]
    df = I * s15[..., None, None] - 3 * d[..., :, None] * d[..., None, :] * s25[..., None, None]
    t = -3 * s25[..., None, None, None]
    ddf = (t * (I[:, :, None] * d[..., None, None, :] + I[:, None, :] * d[..., None, :, None] + I[None, :, :] * d[..., :, None, None])
           + 15 * s35[..., None, None, None] * d[..., :, None, None] * d[..., None, :, None] * d[..., None, None, :])
    return f, df, ddf


def far_expansion(center, src_pos, src_mass, src_ok, eps=EPS, g=1.0):
    """Taylor coefficients (a0, J, Q) about `center` (B,2) of the softened-gravity acceleration from sources
    src_pos (B,M,2) / (M,2), src_mass (B,M) / (M,), weights src_ok (B,M) in {0,1} (1 = source lies outside the window). Direct sum."""
    B = center.shape[0]
    if src_pos.dim() == 2:
        src_pos = src_pos[None].expand(B, -1, -1)
        src_mass = src_mass[None].expand(B, -1)
    w = src_mass * src_ok
    d = src_pos - center[:, None, :]
    f, df, ddf = _fkl(d, eps)
    a0 = g * (w[..., None] * f).sum(1)
    J = -g * (w[..., None, None] * df).sum(1)
    Q = g * (w[..., None, None, None] * ddf).sum(1)
    return a0, J, Q


def cell_grav_acc(pos, mass, eps=EPS, g=1.0, per_cell=16, rings=2, budget=PAIR_BUDGET):
    """Approximate softened-gravity acceleration for one large scene (N,2): direct sum over the (2*rings+1)^2 neighbouring cells,
    monopole+quadrupole cell multipoles (about the cell COM) for all other cells. O(N * (M*(2r+1)^2 + cells))."""
    N = pos.shape[0]
    dev, dt = pos.device, pos.dtype
    lo = pos.min(0).values - 1e-6
    span = (pos.max(0).values - lo).max() * (1 + 1e-9) + 1e-6
    C = int(max(2 * rings + 2, min(96, math.sqrt(N / per_cell))))
    s = span / C
    c2 = ((pos - lo) / s).floor().clamp(0, C - 1).long()
    cid = c2[:, 1] * C + c2[:, 0]
    nc = C * C
    Mc = torch.zeros(nc, device=dev, dtype=dt).index_add_(0, cid, mass)
    com = torch.zeros(nc, 2, device=dev, dtype=dt).index_add_(0, cid, mass[:, None] * pos) / Mc.clamp_min(1e-30)[:, None]
    dl = pos - com[cid]
    qc = torch.zeros(nc, 2, 2, device=dev, dtype=dt).index_add_(0, cid, mass[:, None, None] * dl[:, :, None] * dl[:, None, :])
    order = torch.argsort(cid)
    counts = torch.bincount(cid, minlength=nc)
    starts = torch.cumsum(counts, 0) - counts
    M = int(counts.max())
    cy_all, cx_all = torch.meshgrid(torch.arange(C, device=dev), torch.arange(C, device=dev), indexing="ij")
    cx_all, cy_all = cx_all.reshape(-1), cy_all.reshape(-1)
    sh = torch.arange(-rings, rings + 1, device=dev)
    W = (2 * rings + 1) ** 2
    jj = torch.arange(M, device=dev)
    out = torch.zeros(N, 2, device=dev, dtype=dt)
    Qn = max(1, budget // max(W * M, nc * 8))
    for q0 in range(0, N, Qn):
        q = torch.arange(q0, min(N, q0 + Qn), device=dev)
        x = pos[q]
        cx, cy = c2[q, 0], c2[q, 1]
        far = ((cx_all[None] - cx[:, None]).abs() > rings) | ((cy_all[None] - cy[:, None]).abs() > rings)
        f, _, ddf = _fkl(com[None] - x[:, None, :], eps)
        wf = far.to(dt)
        qq = torch.einsum("ckl,qcikl->qci", qc, ddf)
        a = (wf[..., None] * (Mc[None, :, None] * f + 0.5 * qq)).sum(1)
        nx = (cx[:, None, None] + sh[None, None, :]).expand(-1, 2 * rings + 1, -1).reshape(len(q), W)
        ny = (cy[:, None, None] + sh[None, :, None]).expand(-1, -1, 2 * rings + 1).reshape(len(q), W)
        inside = (nx >= 0) & (nx < C) & (ny >= 0) & (ny < C)
        ncell = ny.clamp(0, C - 1) * C + nx.clamp(0, C - 1)
        cnt = counts[ncell] * inside
        idx = (starts[ncell][..., None] + jj).clamp(max=N - 1)
        valid = jj < cnt[..., None]
        cand = order[idx].reshape(len(q), W * M)
        valid = valid.reshape(len(q), W * M) & (cand != q[:, None])
        d = pos[cand] - x[:, None, :]
        w = torch.where(valid, mass[cand] * ((d ** 2).sum(-1) + eps ** 2) ** -1.5, torch.zeros_like(valid, dtype=dt))
        out[q] = g * (a + (w[..., None] * d).sum(1))
    return out


def grav_acc_large(pos, mass, mask, eps=EPS, g=1.0, direct_max=16384):
    """(B,N,2) softened gravity: exact direct sum for N <= direct_max, else cell multipole (per scene)."""
    if pos.shape[1] <= direct_max:
        return grav_acc(pos, mass, mask, eps, g)
    out = torch.zeros_like(pos)
    for b in range(pos.shape[0]):
        k = mask[b] > 0
        out[b, k] = cell_grav_acc(pos[b, k], mass[b, k], eps, g)
    return out


@torch.no_grad()
def rollout(phys, pos, vel, mass, mask, steps, dt, sub, ext=None, t0=0.0, acc_fn=None):
    """Kick-drift-kick, `sub` substeps per output step dt, under near-field `phys` (or acc_fn(pos,vel,mass,mask)) plus ext.
    Returns dict of (B, steps+1, N, 2): pos, vel, acc_near (target for tier 0), acc_ext. Periodic scenes keep unwrapped positions."""
    h = dt / sub
    near = acc_fn if acc_fn is not None else phys.acc
    ea = (lambda p, t: ext.acc(p, t) * mask[..., None]) if ext is not None else (lambda p, t: torch.zeros_like(p))
    P, V, An, Ae = [], [], [], []
    t = t0
    a_n = near(pos, vel, mass, mask)
    a_e = ea(pos, t)
    for s in range(steps + 1):
        P.append(pos), V.append(vel), An.append(a_n), Ae.append(a_e)
        if s == steps:
            break
        for _ in range(sub):
            vel = vel + 0.5 * h * (a_n + a_e)
            pos = pos + h * vel
            t = t + h
            a_n = near(pos, vel, mass, mask)
            a_e = ea(pos, t)
            vel = vel + 0.5 * h * (a_n + a_e)
    st = lambda L: torch.stack(L, 1)
    return dict(pos=st(P), vel=st(V), acc_near=st(An), acc_ext=st(Ae))


def thermostat_equil(phys, pos, vel, mass, mask, steps, h, temp, tau=0.5):
    """Berendsen equilibration at temperature `temp` (2D, k_B=1) used before recording LJ scenes."""
    a = phys.acc(pos, vel, mass, mask)
    n = mask.sum(-1)
    for _ in range(steps):
        vel = vel + 0.5 * h * a
        pos = pos + h * vel
        a = phys.acc(pos, vel, mass, mask)
        vel = vel + 0.5 * h * a
        T = (mass * mask * (vel ** 2).sum(-1)).sum(-1) / (2 * n)
        vel = vel * torch.sqrt((1 + h / tau * (temp / T.clamp_min(1e-9) - 1)).clamp_min(0.0))[:, None, None]
    return pos, vel
