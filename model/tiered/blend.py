import torch
import torch.nn.functional as F

PHI, GX, GY, HXX, HXY, HYY, RX, RY = range(8)


def smoother(t):
    t = t.clamp(0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t), 30 * t ** 2 * (1 - t) ** 2


class TileLayout:
    """Square GxG tiles over an n x n cell grid of cell size h, each padded by `halo` cells copied from its
    neighbours. Outputs live on the padded tile grid; the blended potential uses a C2 partition of unity over
    cell centres inside each tile and over overlapping tiles (ramp of half-width `blend` cells at each tile edge)."""

    def __init__(self, n, G, halo, blend, h, origin, periodic=False):
        assert n % G == 0
        assert blend == 0 or (halo >= blend + 2 and 2 * blend < G)
        assert halo >= 1
        self.n, self.G, self.halo, self.blend, self.h, self.periodic = n, G, halo, blend, h, periodic
        self.T, self.P = n // G, G + 2 * halo
        self.origin = torch.as_tensor(origin, dtype=torch.float32)

    def extract(self, field):
        hw = self.halo
        x = F.pad(field, (hw,) * 4, mode="circular" if self.periodic else "constant")
        x = x.unfold(2, self.P, self.G).unfold(3, self.P, self.G)
        return x.permute(0, 2, 3, 1, 4, 5).contiguous()

    def interior(self, tiles):
        hw, G = self.halo, self.G
        B, T, _, C = tiles.shape[:4]
        x = tiles[..., hw:hw + G, hw:hw + G]
        return x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, T * G, T * G)

    def centres(self, B, device, dtype):
        c = (torch.arange(self.n, device=device, dtype=dtype) + 0.5) * self.h
        y, x = torch.meshgrid(c, c, indexing="ij")
        p = torch.stack([x, y], -1).reshape(1, -1, 2) + self.origin.to(device=device, dtype=dtype)
        return p.expand(B, -1, -1)

    def _tile_w(self, u, tidx):
        T = self.T
        if self.blend == 0:
            return ((u >= 0) & (u < 1)).to(u.dtype), torch.zeros_like(u)
        aa = self.blend / self.G
        rl, drl = smoother((u + aa) / (2 * aa))
        rr, drr = smoother((u - 1 + aa) / (2 * aa))
        left, dleft, right, dright = rl, drl / (2 * aa), 1 - rr, -drr / (2 * aa)
        if not self.periodic:
            first, last = tidx == 0, tidx == T - 1
            left, dleft = torch.where(first, torch.ones_like(left), left), torch.where(first, torch.zeros_like(left), dleft)
            right, dright = torch.where(last, torch.ones_like(right), right), torch.where(last, torch.zeros_like(right), dright)
        return left * right, dleft * right + left * dright

    def _axis(self, s):
        G, T, hw, h = self.G, self.T, self.halo, self.h
        ar = torch.arange(2, device=s.device)
        cu = s - 0.5
        c0 = torch.floor(cu)
        r, dr = smoother(cu - c0)
        wc = torch.stack([1 - r, r], -1)
        dwc = torch.stack([-dr, dr], -1) / h
        cidx = c0.long()[..., None] + ar
        d = (s[..., None] - (cidx + 0.5)) * h
        tidx = torch.floor(s / G - 0.5).long()[..., None] + ar
        u = s[..., None] / G - tidx
        W, dW = self._tile_w(u, tidx)
        dW = dW / (G * h)
        if self.periodic:
            tstore = tidx % T
        else:
            ok = ((tidx >= 0) & (tidx < T)).to(s.dtype)
            W, dW = W * ok, dW * ok
            tstore = tidx.clamp(0, T - 1)
        p = cidx[..., None, :] - (tidx * G - hw)[..., :, None]
        return W, dW, tstore, wc, dwc, d, p

    def blend_eval(self, coefs, pos):
        """coefs (B,T,T,K,P,P) with K in {1 (scalar field), 6 (phi0,g,H), 8 (+ residual force)}.
        Returns blended value (B,N), its analytic gradient (B,N,2) and blended residual force (B,N,2)."""
        B, N = pos.shape[:2]
        K, P = coefs.shape[3], self.P
        s = (pos - self.origin.to(pos)) / self.h
        if self.periodic:
            s = s % self.n
        ax = [self._axis(s[..., 0]), self._axis(s[..., 1])]
        bidx = torch.arange(B, device=pos.device)[:, None].expand(B, N)
        val = torch.zeros(B, N, device=pos.device, dtype=pos.dtype)
        grad = torch.zeros(B, N, 2, device=pos.device, dtype=pos.dtype)
        res = torch.zeros(B, N, 2, device=pos.device, dtype=pos.dtype)
        (Wx, dWx, tx, wx, dwx, ddx, px), (Wy, dWy, ty, wy, dwy, ddy, py) = ax
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    for e in (0, 1):
                        pxi, pyi = px[..., a, b], py[..., c, e]
                        ok = ((pxi >= 0) & (pxi < P) & (pyi >= 0) & (pyi < P)).to(pos.dtype)
                        co = coefs[bidx, ty[..., c], tx[..., a], :, pyi.clamp(0, P - 1), pxi.clamp(0, P - 1)]
                        ox, oy = Wx[..., a] * wx[..., b], Wy[..., c] * wy[..., e]
                        dox, doy = dWx[..., a] * wx[..., b] + Wx[..., a] * dwx[..., b], dWy[..., c] * wy[..., e] + Wy[..., c] * dwy[..., e]
                        om = ox * oy * ok
                        domx, domy = dox * oy * ok, ox * doy * ok
                        dx, dy = ddx[..., b], ddy[..., e]
                        if K == 1:
                            p = co[..., 0]
                            gx = gy = 0.0
                        else:
                            gx = co[..., GX] + co[..., HXX] * dx + co[..., HXY] * dy
                            gy = co[..., GY] + co[..., HXY] * dx + co[..., HYY] * dy
                            p = co[..., PHI] + co[..., GX] * dx + co[..., GY] * dy + 0.5 * (co[..., HXX] * dx * dx + co[..., HYY] * dy * dy) + co[..., HXY] * dx * dy
                        val = val + om * p
                        grad = grad + torch.stack([domx * p + om * gx, domy * p + om * gy], -1)
                        if K == 8:
                            res = res + om[..., None] * co[..., RX:RY + 1]
        return val, grad, res


def seam_loss(coefs, active, layout):
    """Mean squared difference of the six potential coefficients on cells shared by neighbouring padded tiles."""
    G, hw = layout.G, layout.halo
    c = coefs[:, :, :, :6]
    tot, cnt = c.sum() * 0, 0
    if layout.periodic:
        pairs = [(c[..., G:], c.roll(-1, 2)[..., :2 * hw], active & active.roll(-1, 2), 2),
                 (c[..., G:, :], c.roll(-1, 1)[..., :2 * hw, :], active & active.roll(-1, 1), 1)]
    else:
        pairs = [(c[:, :, :-1, :, :, G:], c[:, :, 1:, :, :, :2 * hw], active[:, :, :-1] & active[:, :, 1:], 2),
                 (c[:, :-1, :, :, G:, :], c[:, 1:, :, :, :2 * hw, :], active[:, :-1] & active[:, 1:], 1)]
    for a, b, both, _ in pairs:
        d = ((a - b) ** 2).mean((3, 4, 5)) * both.to(c.dtype)
        tot = tot + d.sum()
        cnt = cnt + both.sum()
    return tot / cnt.clamp_min(1)


def residual_penalty(coefs, active, h):
    """Soft penalties on the residual force channel: magnitude and discrete curl, over active tiles."""
    r = coefs[:, :, :, RX:RY + 1]
    a = active.to(r.dtype)
    n = a.sum().clamp_min(1)
    mag = ((r ** 2).sum(3).mean((3, 4)) * a).sum() / n
    curl = (r[:, :, :, 1, :-1, 1:] - r[:, :, :, 1, :-1, :-1] - r[:, :, :, 0, 1:, :-1] + r[:, :, :, 0, :-1, :-1]) / h
    return mag, (curl ** 2).mean((3, 4)).mul(a).sum() / n
