"""Radial pair-force kernels for the dual-tree far field: analytic softened gravity, 2D inverse-distance, screened (Yukawa), and the
learned CentralForceDynamics force. A kernel gives the force vector on the target for a source at offset d (pair), and the value and
Jacobian of the force field g(r) for the node-node expansions (g_grad); the Jacobian of a radial force f(|r|) r/|r| is
(f/|r|)(I - rh rh^T) + f'(|r|) rh rh^T, f' by autograd so any scalar f works. `current` is the kernel used by scripts/dual_tree.py and
scripts/dual_estimator.py.
"""
import torch

from model.central_force import CentralForceDynamics


class RadialKernel:
    def __init__(self, f, name):
        self.f, self.name = f, name

    def pair(self, d):
        dn = d.norm(dim=1).clamp(min=1e-9)
        return d * (self.f(dn) / dn)[:, None]

    def g_grad(self, r):
        rn = r.norm(dim=1).clamp(min=1e-9)
        with torch.enable_grad():
            x = rn.detach().requires_grad_(True)
            fx = self.f(x)
            fp = torch.autograd.grad(fx.sum(), x)[0]
        fx = fx.detach()
        rh = r / rn[:, None]
        a = fx / rn
        xx, xy, yy = rh[:, 0] ** 2, rh[:, 0] * rh[:, 1], rh[:, 1] ** 2
        grad = torch.stack([a * (1 - xx) + fp * xx, (fp - a) * xy, (fp - a) * xy, a * (1 - yy) + fp * yy], 1)
        return rh * fx[:, None], grad

    def local_feats(self, d):
        with torch.enable_grad():
            x = d.detach().requires_grad_(True)
            lf = torch.log(self.f(x).abs().clamp(min=1e-30))
            s1 = x * torch.autograd.grad(lf.sum(), x, create_graph=True)[0]
            s2 = x * torch.autograd.grad(s1.sum(), x)[0]
        return torch.stack([s1.detach(), s2.detach()], 1)

    @torch.no_grad()
    def phi(self, d):
        if not hasattr(self, "_tab"):
            g = torch.logspace(-4, 7, 400000, dtype=torch.float64, device=d.device)
            f = torch.cat([self.f(c) for c in g.split(100000)]).double()
            ph = torch.cat([torch.zeros(1, dtype=torch.float64, device=g.device), torch.cumsum(0.5 * (f[1:] + f[:-1]) * (g[1:] - g[:-1]), 0)]) + 0.5 * f[0] * g[0]
            self._tab = (g, ph)
        g, ph = self._tab
        i = torch.searchsorted(g, d.double().clamp(min=g[0], max=g[-1])).clamp(1, len(g) - 1)
        w = (d.double().clamp(min=g[0], max=g[-1]) - g[i - 1]) / (g[i] - g[i - 1])
        return ph[i - 1] * (1 - w) + ph[i] * w

    @torch.no_grad()
    def pair_sums(self, pos, sub=None, chunk=256):
        """(sum_{i<j} phi(d_ij), sum_{i<j} d_ij f(d_ij)) over all pairs of pos (or of a random subsample of `sub` particles, rescaled)."""
        n = pos.shape[0]
        scale = 1.0
        if sub is not None and sub < n:
            pos = pos[torch.randperm(n, device=pos.device)[:sub]]
            scale = n * (n - 1) / (sub * (sub - 1))
        p = pos.double()
        m = p.shape[0]
        pe = w = 0.0
        for i in range(0, m, chunk):
            d = (p[None] - p[i:i + chunk][:, None]).norm(dim=2)
            mask = d > 0
            pe += float(self.phi(d)[mask].sum())
            w += float((d * self.f(d.clamp(min=1e-9)).double())[mask].sum())
        return 0.5 * pe * scale, 0.5 * w * scale

    @torch.no_grad()
    def exact_accel(self, pos, idx, chunk=256):
        src = pos.double()
        out = torch.empty(len(idx), 2, dtype=torch.float64, device=pos.device)
        for i in range(0, len(idx), chunk):
            d = src[None] - src[idx[i:i + chunk]][:, None]
            out[i:i + chunk] = self.pair(d.reshape(-1, 2)).view(d.shape[0], d.shape[1], 2).sum(1)
        return out


def analytic(eps=0.5):
    return RadialKernel(lambda d: d * (d ** 2 + eps ** 2) ** -1.5, "analytic")


def inv_distance(eps=0.5):
    return RadialKernel(lambda d: d / (d ** 2 + eps ** 2), "inv_distance")


def yukawa(eps=0.5, length=30.0):
    return RadialKernel(lambda d: d * (d ** 2 + eps ** 2) ** -1.5 * torch.exp(-d / length), f"yukawa{length:g}")


def learned(ckpt, device, max_d=150.0):
    dyn = CentralForceDynamics(dt=0.1).to(device)
    dyn.load_state_dict(torch.load(ckpt, map_location=device))
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)

    def f(d):
        with torch.enable_grad() if d.requires_grad else torch.no_grad():
            return (dyn.force(torch.log(d.float().clamp(max=max_d))[:, None])[:, 0] / (d.float() ** 2 + 1.0)).to(d.dtype)
    return RadialKernel(f, "learned")


current = analytic()
