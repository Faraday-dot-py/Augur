"""Tiling with arbitrary rectangular cores (opt-in; extends scatter_tiling.TiledScatter, no edits to scatter_field.py).

Each core (multiples of 2 units, width <= 32) is run as one G=128 copy: a window of E=64 units centred on the core, so the halo is >= 16
units for every core. Cores must partition the domain. Ghosts + optional coarse global far term as in TiledScatter; no field exchange
(tile fields move between tilings via `retile`). `split_cores` refines a tiling by body count (quadtree); `lpt_assign` packs tiles on workers.
"""
import torch
import torch.nn.functional as F

from scripts.scatter_tiling import TiledScatter


def uniform_cores(n=4, w=32.0):
    X0 = -n * w / 2
    return [(X0 + i * w, X0 + (i + 1) * w, X0 + j * w, X0 + (j + 1) * w) for i in range(n) for j in range(n)]


def split_cores(cores, pos, thresh, min_w=8.0, dom=128.0):
    """Quadtree refinement by body count: any core holding more than `thresh` bodies and wider than `min_w` is split 2x2, repeatedly."""
    cores = [tuple(c) for c in cores]
    p = pos[(pos.abs() < dom / 2).all(-1)]
    while True:
        out, changed = [], False
        for x0, x1, y0, y1 in cores:
            cnt = int(((p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)).sum())
            if cnt > thresh and (x1 - x0) > min_w:
                xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
                out += [(x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1)]
                changed = True
            else:
                out.append((x0, x1, y0, y1))
        cores = out
        if not changed:
            return cores


def lpt_assign(costs, workers):
    """Longest-processing-time-first packing of tiles onto workers; returns per-worker lists of tile indices."""
    load = [0.0] * workers
    bins = [[] for _ in range(workers)]
    for k in sorted(range(len(costs)), key=lambda i: -costs[i]):
        w = min(range(workers), key=lambda i: load[i])
        bins[w].append(k)
        load[w] += costs[k]
    return bins


class CoreTiled(TiledScatter):
    def __init__(self, model, cores, domain=128.0, E=64.0, far=True, far_c=4):
        self.m = model
        self.h = float(model.h)
        self.E, self.G, self.D, self.X0 = float(E), round(E / self.h), float(domain), -domain / 2
        self.far, self.c = far, far_c
        self.ghosts, self.field_x, self.H = True, False, 0.0
        self.stats = {}
        self.set_cores(cores)

    def set_cores(self, cores):
        cs = self.c * self.h
        self.cores = [tuple(float(v) for v in c) for c in cores]
        self.T = len(cores)
        t = torch.tensor(self.cores, dtype=torch.float64)
        ctr = torch.stack([(t[:, 0] + t[:, 1]) / 2, (t[:, 2] + t[:, 3]) / 2], 1)
        self.w0 = torch.round((ctr - self.E / 2) / cs) * cs
        assert ((t[:, 1] - t[:, 0]) <= 32).all() and ((t[:, 3] - t[:, 2]) <= 32).all()
        n = round(self.D / cs)
        idmap = torch.full((n, n), -1, dtype=torch.long)
        for k, (x0, x1, y0, y1) in enumerate(self.cores):
            idmap[round((y0 - self.X0) / cs):round((y1 - self.X0) / cs), round((x0 - self.X0) / cs):round((x1 - self.X0) / cs)] = k
        assert (idmap >= 0).all(), "cores must partition the domain"
        self.idmap_cpu, self.idmap, self.n_coarse = idmap, None, n

    def _dev(self, dev):
        if self.idmap is None or self.idmap.device != dev:
            self.idmap = self.idmap_cpu.to(dev)
            self.w0_dev = self.w0.to(dev)

    def cid(self, pos):
        self._dev(pos.device)
        u = ((pos - self.X0) / (self.c * self.h)).floor().long().clamp(0, self.n_coarse - 1)
        return self.idmap[u[:, 1], u[:, 0]]

    def init_field(self, dev, dtype=torch.float32):
        return torch.zeros(self.T, self.m.cf, self.G, self.G, device=dev, dtype=dtype)

    def assemble_fields(self, f):
        h, X0 = self.h, self.X0
        nd = round(self.D / h)
        g = f.new_zeros(f.shape[1], nd, nd)
        for k, (x0, x1, y0, y1) in enumerate(self.cores):
            wx, wy = float(self.w0[k, 0]), float(self.w0[k, 1])
            g[:, round((y0 - X0) / h):round((y1 - X0) / h), round((x0 - X0) / h):round((x1 - X0) / h)] = \
                f[k, :, round((y0 - wy) / h):round((y1 - wy) / h), round((x0 - wx) / h):round((x1 - wx) / h)]
        return g

    def slice_fields(self, glob):
        pad = self.G // 2
        p = F.pad(glob, (pad,) * 4)
        out = []
        for k in range(self.T):
            ix = round((float(self.w0[k, 0]) - self.X0) / self.h) + pad
            iy = round((float(self.w0[k, 1]) - self.X0) / self.h) + pad
            out.append(p[:, iy:iy + self.G, ix:ix + self.G])
        return torch.stack(out)

    def retile(self, cores, field):
        glob = self.assemble_fields(field)
        self.set_cores(cores)
        return self.slice_fields(glob)

    def _pack(self, pos, vel, mass, tiles):
        cid = self.cid(pos)
        inside = (pos.abs() < self.D / 2 + self.h / 2).all(-1)
        idx, nown = [], []
        for k in tiles:
            own = (cid == k).nonzero()[:, 0]
            lo = self.w0_dev[k].to(pos.dtype)
            hi = lo + self.E
            g = ((cid != k) & inside & (pos >= lo).all(-1) & (pos < hi).all(-1)).nonzero()[:, 0]
            idx.append(torch.cat([own, g]))
            nown.append(len(own))
        Nmax = max(2, max(len(i) for i in idx))
        T, dev = len(tiles), pos.device
        P = torch.zeros(T, Nmax, 2, device=dev, dtype=pos.dtype)
        V = torch.zeros_like(P)
        M, K, OWN, INS = [torch.zeros(T, Nmax, device=dev, dtype=pos.dtype) for _ in range(4)]
        for j, k in enumerate(tiles):
            c = self.w0_dev[k].to(pos.dtype) + self.E / 2
            i = idx[j]
            P[j, :len(i)] = pos[i] - c
            V[j, :len(i)] = vel[i]
            M[j, :len(i)] = mass[i]
            K[j, :len(i)] = 1.0
            OWN[j, :nown[j]] = 1.0
            INS[j, :len(i)] = inside[i].to(pos.dtype)
        self.stats = {"own": nown, "ghost": [len(i) - o for i, o in zip(idx, nown)], "nmax": Nmax}
        return P, V, M, K, OWN, INS, idx

    def _far_acc(self, rho):
        c, hc = self.c, self.h * self.c
        rc = F.avg_pool2d(rho, c)
        wc = rc.shape[-1]
        n, pad = self.n_coarse, 16
        Rg = rc.new_zeros(1, n, n)
        for k, (x0, x1, y0, y1) in enumerate(self.cores):
            wx, wy = float(self.w0[k, 0]), float(self.w0[k, 1])
            Rg[:, round((y0 - self.X0) / hc):round((y1 - self.X0) / hc), round((x0 - self.X0) / hc):round((x1 - self.X0) / hc)] = \
                rc[k, :, round((y0 - wy) / hc):round((y1 - wy) / hc), round((x0 - wx) / hc):round((x1 - wx) / hc)]
        Rp = F.pad(Rg, (pad,) * 4)[None]
        phig = self._phi(Rp, Rp.shape[-1], hc)[0]
        win = []
        for k in range(self.T):
            ix = round((float(self.w0[k, 0]) - self.X0) / hc) + pad
            iy = round((float(self.w0[k, 1]) - self.X0) / hc) + pad
            win.append(phig[:, iy:iy + wc, ix:ix + wc])
        far = torch.stack(win) - self._phi(rc, wc, hc)
        far = F.interpolate(far, scale_factor=c, mode="bilinear", align_corners=False)
        return self.m.neg_grad(far)

    def seam_dist(self, pos):
        """distance to the nearest internal boundary of the body's own core (inf if none) and to the outer domain boundary."""
        cs = torch.tensor(self.cores, device=pos.device, dtype=pos.dtype)[self.cid(pos)]
        lo, hi = self.X0, -self.X0
        big = torch.full_like(pos[:, 0], float("inf"))
        d = [torch.where(cs[:, a] > lo + 1e-6, pos[:, ax] - cs[:, a], big) for ax, a in ((0, 0), (1, 2))]
        d += [torch.where(cs[:, a] < hi - 1e-6, cs[:, a] - pos[:, ax], big) for ax, a in ((0, 1), (1, 3))]
        return torch.stack(d, 0).amin(0), (self.D / 2 - pos.abs()).min(-1).values

    def force(self, pos, vel, mass, field, tiles=None):
        m = self.m
        tiles = list(range(self.T)) if tiles is None else list(tiles)
        far = self.far and len(tiles) == self.T
        old = self._scope()
        try:
            P, V, M, K, OWN, INS, idx = self._pack(pos, vel, mass, tiles)
            x = m.scatter(P, V, M, K * INS)
            ak = m.kernel_acc(x[:, :1])
            if far:
                ak = ak + self._far_acc(x[:, :1])
            xin = torch.cat([x, ak, field[tiles]], 1)
            f = x.new_zeros(len(tiles), m.cf, self.G, self.G) if m.nonet else m.net(xin)
            acc = m.neg_grad(f[:, :1]) + ak
            kw = {"qmask": OWN} if "adapt" in m.opt else {}
            a = m.gather(acc, P) * INS[..., None] + m.pp_acc(P, M, K, **kw)
            a = a * K[..., None]
        finally:
            self._restore(old)
        out = torch.zeros_like(pos)
        for j, i in enumerate(idx):
            no = int(OWN[j].sum())
            out[i[:no]] = a[j, :no]
        mm = mass[:, None]
        out = out - (mm * out).sum(0, keepdim=True) / mm.sum()
        newf = field.clone()
        newf[tiles] = f
        return out, newf
