"""Opt-in obstacle + ball-ball contact extension of ScatterField for static solid obstacles of arbitrary shape.
See docs/debugging/obstacle-contact-force-options.md (options A/B) and scatter-contact.md.

Obstacles are SOLID regions rasterised to a signed distance field (SDF, positive in free space) on the model grid. Primitives:
  ("hbox", lo(2), hi(2))  free space inside an axis-aligned rectangle (solid outside)
  ("hdisc", c(2), r)      free space inside a disc
  ("circle", c(2), r)     solid disc
  ("poly", pts(E,2))      solid simple polygon
obstacle="sdf" (option B): SDF + normal gathered at each ball, a 5-point stencil (p, p+-s n, p+-s t) and (v.n, v.t, material) go
into an MLP whose output is a force in the (n, t) frame. obstacle="kernel" (option A): phi = K * occupancy by FFT, force = -grad phi.
Both add the force AFTER momfix. Ball-ball contact: radius-aware pair MLP on the overlap gap r-(Ri+Rj) over k nearest neighbours (cell list for large N).
All new heads are zero-initialised (gscale=1), so an untrained ContactScatterField loaded from a ScatterField checkpoint reproduces it exactly.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import scatter_field as sf

OUT_SCALE = 10.0
FMAX = 40.0


def bound(x):
    return FMAX * torch.tanh(x * (OUT_SCALE / FMAX))


def hbox(lo, hi):
    return dict(kind="hbox", lo=tuple(float(x) for x in lo), hi=tuple(float(x) for x in hi))


def hdisc(c, r):
    return dict(kind="hdisc", c=tuple(float(x) for x in c), r=float(r))


def circle(c, r):
    return dict(kind="circle", c=tuple(float(x) for x in c), r=float(r))


def poly(pts):
    return dict(kind="poly", pts=[tuple(float(x) for x in p) for p in pts])


def regular_poly(c, r, n, rot):
    a = rot + 2 * math.pi * np.arange(n) / n
    return poly(np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1))


def rect_poly(c, w, h, rot):
    cs, sn = math.cos(rot), math.sin(rot)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return poly([(c[0] + cs * x - sn * y, c[1] + sn * x + cs * y) for x, y in corners])


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def prim_terms(pr, p):
    """List of (d, n): signed distance (positive in free space) and its unit gradient. A hbox gives its four half-planes."""
    k = pr["kind"]
    if k == "hbox":
        lo, hi = pr["lo"], pr["hi"]
        z, o = torch.zeros_like(p[..., 0]), torch.ones_like(p[..., 0])
        ex, ey = torch.stack([o, z], -1), torch.stack([z, o], -1)
        return [(p[..., 0] - lo[0], ex), (hi[0] - p[..., 0], -ex), (p[..., 1] - lo[1], ey), (hi[1] - p[..., 1], -ey)]
    if k in ("circle", "hdisc"):
        v = p - torch.tensor(pr["c"], dtype=p.dtype, device=p.device)
        dist = v.norm(dim=-1)
        n = v / dist.clamp(min=1e-12)[..., None]
        return [(dist - pr["r"], n)] if k == "circle" else [(pr["r"] - dist, -n)]
    pts = torch.tensor(pr["pts"], dtype=p.dtype, device=p.device)
    a, b = pts, torch.roll(pts, -1, 0)
    e = b - a
    q = p[..., None, :]
    t = (((q - a) * e).sum(-1) / (e * e).sum(-1)).clamp(0, 1)
    cp = a + t[..., None] * e
    d2 = ((q - cp) ** 2).sum(-1)
    i = d2.argmin(-1)
    dist = torch.sqrt(d2.gather(-1, i[..., None])[..., 0])
    cpi = cp.gather(-2, i[..., None, None].expand(*i.shape, 1, 2))[..., 0, :]
    ay, by = a[:, 1], b[:, 1]
    cross = ((ay > q[..., 1]) != (by > q[..., 1])) & (q[..., 0] < (b[:, 0] - a[:, 0]) * (q[..., 1] - ay) / (by - ay + 1e-30) + a[:, 0])
    inside = cross.sum(-1) % 2 == 1
    sgn = torch.where(inside, -torch.ones_like(dist), torch.ones_like(dist))
    n = sgn[..., None] * _unit(p - cpi)
    return [(sgn * dist, n)]


def prim_sdf(pr, p):
    ts = prim_terms(pr, p)
    return torch.stack([d for d, _ in ts], 0).min(0).values


def scene_sdf(prims, p):
    return torch.stack([prim_sdf(pr, p) for pr in prims], 0).min(0).values


def rasterize(prims, grid, extent, mat=0.0):
    """-> float32 [4, G, G]: sdf, d(sdf)/dx, d(sdf)/dy (per unit length), material (reserved slot)."""
    h = extent / grid
    xs = (torch.arange(grid, dtype=torch.float64) + 0.5) * h - extent / 2
    p = torch.stack([xs[None, :].expand(grid, grid), xs[:, None].expand(grid, grid)], -1)
    s = scene_sdf(prims, p)
    gy, gx = torch.gradient(s, spacing=h)
    return torch.stack([s, gx, gy, torch.full_like(s, mat)], 0).float()


def occupancy(geom, h):
    return (0.5 - geom[:, :1] / h).clamp(0, 1)


def contact_truth(prims, pos0, vel0, rad, steps, dev, dt=0.15, sub=8, g=9.0, k=400.0):
    """fp64 penalty-force physics (bounce.py convention, symplectic Euler) with SDF walls and per-body radii. Gravity (g, 0).
    Each primitive term contributes k*(R-d)^2 along its normal; ball pairs k*(Ri+Rj-r)^2. Returns numpy (T+1,N,2) pos, vel."""
    p = torch.tensor(pos0, dtype=torch.float64, device=dev)
    v = torch.tensor(vel0, dtype=torch.float64, device=dev)
    R = torch.tensor(rad, dtype=torch.float64, device=dev)
    n = p.shape[0]
    RR = R[:, None] + R[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    terms = [t for pr in prims for t in [pr]]
    h = dt / sub

    def force(p):
        f = torch.zeros_like(p)
        f[:, 0] = g
        for pr in terms:
            for d, nn_ in prim_terms(pr, p):
                x = (R - d).clamp(min=0)
                f = f + (k * x * x)[:, None] * nn_
        d = p[:, None, :] - p[None, :, :]
        dist = d.norm(dim=-1)
        x = (RR - dist).clamp(min=0).masked_fill(eye, 0.0)
        return f + ((k * x * x / dist.clamp(min=1e-9))[..., None] * d).sum(1)

    ps, vs = [p.cpu().numpy()], [v.cpu().numpy()]
    for _ in range(steps):
        for _ in range(sub):
            v = v + force(p) * h
            p = p + v * h
        ps.append(p.cpu().numpy())
        vs.append(v.cpu().numpy())
    return np.stack(ps), np.stack(vs)


def knn_dense(pos, mask, k):
    """All-pairs k nearest valid neighbours. Returns ds [B*N,k,2] (neighbour - self), r [B*N,k] (1e5 if invalid), nb [B*N,k] flat neighbour index."""
    B, N = mask.shape
    k = min(k, N - 1)
    d_all = pos[:, None, :, :] - pos[:, :, None, :]
    r_all = torch.sqrt((d_all ** 2).sum(-1) + 1e-8)
    bad = (1 - mask)[:, None, :] + torch.eye(N, device=pos.device, dtype=pos.dtype)[None]
    r_sel, idx = torch.topk(r_all + 1e6 * (bad > 0).to(pos.dtype), k, dim=-1, largest=False)
    ds = torch.gather(d_all, 2, idx[..., None].expand(B, N, k, 2))
    nb = idx + (torch.arange(B, device=pos.device) * N)[:, None, None]
    return ds.reshape(B * N, k, 2), r_sel.reshape(B * N, k), nb.reshape(B * N, k)


def knn_cell(pos, mask, k, cell, extent, budget=2 ** 25):
    """Same output as knn_dense through a uniform cell list (cell >= largest wanted neighbour distance for exactness)."""
    B, N = mask.shape
    dev = pos.device
    C = int(math.ceil(extent / cell))
    pf, mk = pos.reshape(B * N, 2), mask.reshape(-1)
    c = ((pf + extent / 2) / cell).floor().clamp(0, C - 1).long().detach()
    b = torch.arange(B, device=dev).repeat_interleave(N)
    nc = B * C * C
    key = torch.where(mk > 0, b * C * C + c[:, 1] * C + c[:, 0], torch.full_like(b, nc))
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=nc + 1)
    starts = torch.cumsum(counts, 0) - counts
    M = int(counts[:nc].max())
    sh = torch.tensor([-1, 0, 1], device=dev)
    k = min(k, 9 * M)
    j = torch.arange(M, device=dev)
    outs = []
    Q = max(1, budget // (9 * M))
    for q0 in range(0, B * N, Q):
        q = torch.arange(q0, min(q0 + Q, B * N), device=dev)
        ncx = (c[q, 0, None] + sh)[:, None, :].expand(-1, 3, -1)
        ncy = (c[q, 1, None] + sh)[:, :, None].expand(-1, -1, 3)
        inside = ((ncx >= 0) & (ncx < C) & (ncy >= 0) & (ncy < C)).reshape(len(q), 9)
        ncell = b[q, None] * C * C + ncy.clamp(0, C - 1).reshape(len(q), 9) * C + ncx.clamp(0, C - 1).reshape(len(q), 9)
        cnt = counts[ncell] * inside
        idx = (starts[ncell][..., None] + j).clamp(max=B * N - 1)
        valid = j < cnt[..., None]
        cand = order[idx].reshape(len(q), 9 * M)
        bad = (~valid.reshape(len(q), 9 * M)) | (cand == q[:, None]) | (mk[q][:, None] == 0)
        d = pf[cand] - pf[q][:, None]
        r_all = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        r_sel, i = torch.topk(r_all + 1e6 * bad.to(pos.dtype), k, dim=-1, largest=False)
        outs.append((torch.gather(d, 1, i[..., None].expand(-1, -1, 2)), r_sel, torch.gather(cand, 1, i)))
    return tuple(torch.cat(x) for x in zip(*outs))


def _mlp(cin, cout):
    m = nn.Sequential(nn.Linear(cin, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, cout))
    nn.init.zeros_(m[-1].weight)
    nn.init.zeros_(m[-1].bias)
    return m


def _win(x, rng):
    return (1 - (x.clamp(min=0) / rng).clamp(max=1.0) ** 2) ** 2


class ContactScatterField(sf.ScatterField):
    def __init__(self, *args, obstacle="sdf", pw=2.0, pc=2.0, stencil=1.0, knn=8, r0=0.75, cell=4.0, kernel_range=4.0, **kw):
        super().__init__(*args, **kw)
        assert obstacle in ("sdf", "kernel", "none")
        self.obstacle, self.pw, self.pc, self.stencil, self.knn, self.r0, self.cell, self.kr = obstacle, pw, pc, stencil, knn, r0, cell, kernel_range
        self.gvec = nn.Parameter(torch.zeros(2))
        self.gscale = nn.Parameter(torch.ones(()))
        self.cmlp = _mlp(2, 1)
        if obstacle == "sdf":
            self.wmlp = _mlp(9, 2)
        elif obstacle == "kernel":
            self.wkmlp = _mlp(2, 1)
        self.geom = None
        self.rad = None

    def set_scene(self, geom=None, rad=None):
        """geom [B or 1, 4, G, G] from rasterize (None = no obstacles); rad [B,N] or scalar per-body radius (None = r0)."""
        self.geom, self.rad = geom, rad

    def _rad(self, B, N, ref):
        if self.rad is None:
            return torch.full((B, N), self.r0, device=ref.device, dtype=ref.dtype)
        r = torch.as_tensor(self.rad, device=ref.device, dtype=ref.dtype)
        return r.expand(B, N) if r.dim() < 2 else r

    def _gather_geom(self, geom, pts):
        B = pts.shape[0]
        if geom.shape[0] != B:
            geom = geom.expand(B, -1, -1, -1)
        out = F.grid_sample(geom, (pts / (self.extent / 2))[:, :, None, :], mode="bilinear", padding_mode="border", align_corners=False)
        return out[..., 0].permute(0, 2, 1)

    def wall_acc_sdf(self, pos, vel, rad):
        B, N = pos.shape[:2]
        g0 = self._gather_geom(self.geom, pos)
        d0, n = g0[..., 0], _unit(g0[..., 1:3])
        t = torch.stack([-n[..., 1], n[..., 0]], -1)
        s = self.stencil
        pts = torch.cat([pos + s * n, pos - s * n, pos + s * t, pos - s * t], 1)
        ds = (self._gather_geom(self.geom, pts)[..., 0].reshape(B, 4, N).permute(0, 2, 1) - d0[..., None]) / s
        gap = d0 - rad
        feat = torch.cat([gap.clamp(min=-3.0, max=8.0)[..., None], torch.log(gap.clamp(min=0.0) + 0.05)[..., None],
                          ((vel * n).sum(-1) / 10.0)[..., None], ((vel * t).sum(-1) / 10.0)[..., None], ds.clamp(-2, 2), g0[..., 3:4]], -1)
        o = bound(self.wmlp(feat)) * _win(gap, self.pw)[..., None]
        return o[..., :1] * n + o[..., 1:] * t

    def wall_acc_kernel(self, pos):
        G, h = self.grid, self.h
        occ = occupancy(self.geom, h)
        idx = torch.arange(2 * G, device=occ.device, dtype=occ.dtype)
        d = torch.minimum(idx, 2 * G - idx) * h
        r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        K = self.wkmlp(torch.stack([r, torch.log(r + h)], -1))[..., 0] * _win(r, self.kr)
        phi = torch.fft.irfft2(torch.fft.rfft2(occ, s=(2 * G, 2 * G)) * torch.fft.rfft2(K), s=(2 * G, 2 * G))[..., :G, :G] * h ** 2
        a = self.neg_grad(phi)
        if a.shape[0] != pos.shape[0]:
            a = a.expand(pos.shape[0], -1, -1, -1).contiguous()
        return bound(self.gather(a, pos))

    def pair_acc(self, pos, mask, rad):
        B, N = mask.shape
        if N < 2:
            return torch.zeros_like(pos)
        if N > 512:
            ds, r, nb = knn_cell(pos, mask, self.knn, self.cell, self.extent)
        else:
            ds, r, nb = knn_dense(pos, mask, self.knn)
        rf = rad.reshape(-1)
        ok = (r < 1e5).to(pos.dtype) * mask.reshape(-1)[:, None]
        r = r.clamp(max=1e5)
        gap = r - (rf[:, None] + rf[nb])
        feat = torch.stack([gap.clamp(max=8.0), torch.log(gap.clamp(min=0.0) + 0.05)], -1)
        g = bound(self.cmlp(feat)[..., 0]) * _win(gap, self.pc) * ok
        return ((g[..., None] * ds / r[..., None]).sum(1)).reshape(B, N, 2)

    def local_acc(self, pos, vel, mask):
        B, N = mask.shape
        rad = self._rad(B, N, pos)
        a = OUT_SCALE * self.gvec + self.pair_acc(pos, mask, rad)
        if self.geom is not None and self.obstacle == "sdf":
            a = a + self.wall_acc_sdf(pos, vel, rad)
        elif self.geom is not None and self.obstacle == "kernel":
            a = a + self.wall_acc_kernel(pos)
        return a * mask[..., None]

    def contact_gate(self, pos, mask):
        """(B,N) in [0,1]: 1 in contact, 0 beyond `pw` of any wall or neighbour gap (same window as the learned contact terms)."""
        B, N = mask.shape
        rad = self._rad(B, N, pos)
        gap = torch.full((B, N), 1e3, device=pos.device, dtype=pos.dtype)
        if self.geom is not None and self.obstacle == "sdf":
            gap = torch.minimum(gap, self._gather_geom(self.geom, pos)[..., 0] - rad)
        if N > 1:
            ds, r, nb = knn_dense(pos, mask, 1)
            rf = rad.reshape(-1)
            pg = (r[:, 0].clamp(max=1e5) - (rf + rf[nb[:, 0]])).reshape(B, N)
            gap = torch.minimum(gap, pg)
        return _win(gap, self.pw) * mask

    def _force(self, pos, vel, mass, mask, field):
        if self.t0 is not None and self.t0.cfg.head and self.t0.cfg.headgate:
            self.t0.gate = self.contact_gate(pos, mask)
        a, field = super()._force(pos, vel, mass, mask, field)
        return a * self.gscale + self.local_acc(pos, vel, mask), field
