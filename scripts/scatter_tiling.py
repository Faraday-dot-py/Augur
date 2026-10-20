"""Tiling of ScatterField copies (opt-in; no edits to scatter_field.py).

nt x nt tiles, each owning a square of side `tile` and running one copy of the model on a grid covering its square plus a
`halo` margin (extent tile + 2 halo, cell size unchanged, so G = (tile + 2 halo) / h; G must be a multiple of 32).
Mechanisms (each a flag):
  ghosts   bodies of neighbouring tiles lying inside this tile's extended box are deposited and feed the pair term; only the
           tile's own bodies keep their force. Without it a tile sees only its own bodies (naive tiling, no communication).
  field_x  after each force the halo cells of the recurrent potential field are overwritten by the neighbours' interior values.
           Without it every tile keeps its own halo field.
  far      far-field coupling: a coarse global grid (cell far_c * h) carries the learned kernel convolution of ALL mass; each tile
           adds (coarse global potential - coarse potential of its own extended box), i.e. the kernel contribution of mass outside
           its box, to the kernel acceleration input and output.
Momentum fix (mass-weighted mean acceleration removal) is done globally (two scalars all-reduced).
All tiles run as one batch on one device (emulation of one device per tile); communication is exact array copies.
"""
import torch
import torch.nn.functional as F


class TiledScatter:
    def __init__(self, model, nt, tile, halo, ghosts=True, field_x=True, far=False, far_c=4):
        self.m = model
        self.nt, self.Lt, self.H = nt, float(tile), float(halo)
        self.h = float(model.h)
        self.E = self.Lt + 2 * self.H
        self.G = round(self.E / self.h)
        self.gt = round(self.Lt / self.h)
        self.hg = round(self.H / self.h)
        assert self.G % 32 == 0, "extended grid must be a multiple of 32 cells"
        self.X0 = -nt * self.Lt / 2
        self.ghosts, self.field_x, self.far, self.c = ghosts, field_x, far, far_c
        if far:
            assert self.gt % far_c == 0 and self.hg % far_c == 0
        self.stats = {}

    def _scope(self):
        m = self.m
        old = (m.grid, m.extent, m.h, m.momfix)
        m.grid, m.extent, m.h, m.momfix = self.G, self.E, self.h, False
        return old

    def _restore(self, old):
        m = self.m
        m.grid, m.extent, m.h, m.momfix = old

    def init_field(self, dev, dtype=torch.float32):
        return torch.zeros(self.nt ** 2, self.m.cf, self.G, self.G, device=dev, dtype=dtype)

    def slice_field(self, glob):
        """glob (C, nt*gt, nt*gt) global field -> per-tile extended fields (T, C, G, G); zeros outside the domain."""
        p = F.pad(glob, (self.hg,) * 4)
        out = []
        for tx in range(self.nt):
            for ty in range(self.nt):
                out.append(p[:, ty * self.gt:ty * self.gt + self.G, tx * self.gt:tx * self.gt + self.G])
        return torch.stack(out)

    def assemble(self, loc, c=1):
        """per-tile (T, C, G, G) -> global (C, nt*gt, nt*gt) from tile interiors (c = block factor for coarse arrays)."""
        gt, hg, n = self.gt // c, self.hg // c, self.nt
        g = loc.new_zeros(loc.shape[1], n * gt, n * gt)
        for tx in range(n):
            for ty in range(n):
                g[:, ty * gt:(ty + 1) * gt, tx * gt:(tx + 1) * gt] = loc[tx * n + ty, :, hg:hg + gt, hg:hg + gt]
        return g

    def _pack(self, pos, vel, mass, tiles):
        n = self.nt
        t = ((pos - self.X0) / self.Lt).floor().clamp(0, n - 1).long()
        tid = t[:, 0] * n + t[:, 1]
        inside = (pos.abs() < n * self.Lt / 2 + self.h / 2).all(-1)
        idx, nown = [], []
        for k in tiles:
            tx, ty = divmod(k, n)
            own = (tid == k).nonzero()[:, 0]
            if self.ghosts and self.H > 0:
                lo = torch.tensor([self.X0 + tx * self.Lt - self.H, self.X0 + ty * self.Lt - self.H], device=pos.device)
                hi = lo + self.E
                g = ((tid != k) & inside & (pos >= lo).all(-1) & (pos < hi).all(-1)).nonzero()[:, 0]
            else:
                g = own[:0]
            idx.append(torch.cat([own, g]))
            nown.append(len(own))
        Nmax = max(2, max(len(i) for i in idx))
        T = len(tiles)
        dev = pos.device
        P = torch.zeros(T, Nmax, 2, device=dev, dtype=pos.dtype)
        V = torch.zeros_like(P)
        M = torch.zeros(T, Nmax, device=dev, dtype=pos.dtype)
        K = torch.zeros_like(M)
        OWN = torch.zeros_like(M)
        INS = torch.zeros_like(M)
        for j, k in enumerate(tiles):
            tx, ty = divmod(k, n)
            c = torch.tensor([self.X0 + (tx + 0.5) * self.Lt, self.X0 + (ty + 0.5) * self.Lt], device=dev, dtype=pos.dtype)
            i = idx[j]
            P[j, :len(i)] = pos[i] - c
            V[j, :len(i)] = vel[i]
            M[j, :len(i)] = mass[i]
            K[j, :len(i)] = 1.0
            OWN[j, :nown[j]] = 1.0
            INS[j, :len(i)] = inside[i].to(pos.dtype)
        self.stats = {"ghosts_mean": float(sum(len(i) - o for i, o in zip(idx, nown))) / T, "own_mean": float(sum(nown)) / T,
                      "nmax": Nmax}
        return P, V, M, K, OWN, INS, idx

    def _phi(self, rho, G, h):
        m = self.m
        old = (m.grid, m.h)
        m.grid, m.h = G, h
        Kf = m._kernel_fft(rho)
        m.grid, m.h = old
        rf = torch.fft.rfft2(rho, s=(2 * G, 2 * G))
        return torch.fft.irfft2(rf * Kf, s=(2 * G, 2 * G))[..., :G, :G] * h ** 2

    def _far_acc(self, rho):
        c, hc = self.c, self.h * self.c
        rc = F.avg_pool2d(rho, c)
        n, gtc, hgc = self.nt, self.gt // c, self.hg // c
        Rg = rc.new_zeros(1, n * gtc, n * gtc)
        for tx in range(n):
            for ty in range(n):
                Rg[:, ty * gtc:(ty + 1) * gtc, tx * gtc:(tx + 1) * gtc] = rc[tx * n + ty, :, hgc:hgc + gtc, hgc:hgc + gtc]
        Rp = F.pad(Rg, (hgc,) * 4)[None]
        Gp = Rp.shape[-1]
        phig = self._phi(Rp, Gp, hc)[0]
        win = torch.stack([phig[:, ty * gtc:ty * gtc + rc.shape[-1], tx * gtc:tx * gtc + rc.shape[-1]]
                           for tx in range(n) for ty in range(n)])
        far = win - self._phi(rc, rc.shape[-1], hc)
        far = F.interpolate(far, scale_factor=c, mode="bilinear", align_corners=False)
        return self.m.neg_grad(far)

    def force(self, pos, vel, mass, field, tiles=None):
        """pos/vel (N,2), mass (N,), field (T,C,G,G) tile-local. Returns acceleration (N,2) and the next tile fields."""
        m = self.m
        T = self.nt ** 2
        tiles = list(range(T)) if tiles is None else list(tiles)
        assert not self.far or len(tiles) == T
        old = self._scope()
        try:
            P, V, M, K, OWN, INS, idx = self._pack(pos, vel, mass, tiles)
            x = m.scatter(P, V, M, K * INS)
            ak = m.kernel_acc(x[:, :1])
            if self.far:
                ak = ak + self._far_acc(x[:, :1])
            xin = torch.cat([x, ak, field[tiles]], 1)
            f = x.new_zeros(len(tiles), m.cf, self.G, self.G) if m.nonet else m.net(xin)
            acc = m.neg_grad(f[:, :1]) + ak
            a = m.gather(acc, P) * INS[..., None] + m.pp_acc(P, M, K)
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
        if self.field_x and self.H > 0 and len(tiles) == T:
            newf = self.slice_field(self.assemble(f))
        return out, newf

    def step(self, pos, vel, mass, st):
        field, a_prev = st
        dt = self.m.dt
        if a_prev is None:
            a_prev, field = self.force(pos, vel, mass, field)
        pn = pos + dt * vel + 0.5 * dt ** 2 * a_prev
        a_new, field = self.force(pn, vel + dt * a_prev, mass, field)
        dv = 0.5 * dt * (a_prev + a_new)
        return pn, vel + dv, (field, a_new)

    def seam_dist(self, pos):
        """distance to the nearest internal tile boundary (inf if nt == 1) and to the outer domain boundary."""
        L = self.nt * self.Lt
        d_out = (L / 2 - pos.abs()).min(-1).values
        if self.nt == 1:
            return torch.full_like(d_out, float("inf")), d_out
        b = self.X0 + self.Lt * torch.arange(1, self.nt, device=pos.device, dtype=pos.dtype)
        ds = (pos[..., None] - b).abs().amin((-1, -2))
        return ds, d_out
