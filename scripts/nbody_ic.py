"""Initial conditions for N-body rollout tests (2D, unit masses, seed 4738, GPU). Lengths scale as s = sqrt(N/1e5) so the density regime
matches the 100k flyby (cluster sigma 60 at N=100k); virial dispersion from the kernel via 2K = sum_{i<j} d f(d) (subsampled).
Each returns (pos, vel) float64 with zero net momentum. Names: uniform, flyby, three, plummer, disk, clumpy.
"""
import math

import torch

from scripts import gravity_1b as g1


def _gen(dev, seed):
    return torch.Generator(device=dev).manual_seed(seed)


def virial_sigma(pos, kernel, ratio=0.5, sub=6000):
    _, w = kernel.pair_sums(pos, sub=sub)
    return math.sqrt(ratio * w / pos.shape[0])


def _cluster(n, centre, sigma, kernel, gen, dev, ratio=0.5, dim=None):
    dim = dim if dim is not None else len(centre)
    pos = torch.randn(n, dim, generator=gen, device=dev, dtype=torch.float64) * sigma + torch.tensor(centre, device=dev, dtype=torch.float64)
    sv = virial_sigma(pos, kernel, ratio)
    return pos, torch.randn(n, dim, generator=gen, device=dev, dtype=torch.float64) * sv


def _finish(pos, vel):
    return pos, vel - vel.mean(0)


def uniform(n, kernel, dev, seed=4738):
    pos, vel, _ = g1.init_state(n, dev, seed)
    return _finish(pos.double(), vel.double())


def flyby(n, kernel, dev, seed=4738):
    gen, s, v = _gen(dev, seed), math.sqrt(n / 1e5), (n / 1e5) ** 0.25
    sigma, sep, imp, app, spin = 60 * s, 800 * s, 240 * s, 11 * v, 9 * v
    ca, cb = (-sep / 2, -imp / 2), (sep / 2, imp / 2)
    out = []
    for c, sgn in ((ca, 1.0), (cb, -1.0)):
        p, vl = _cluster(n // 2, c, sigma, kernel, gen, dev)
        rel = p - torch.tensor(c, device=dev, dtype=torch.float64)
        vl = vl + spin / sigma * torch.stack([-rel[:, 1], rel[:, 0]], 1)
        vl[:, 0] += sgn * app
        out.append((p, vl))
    return _finish(torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]))


def three(n, kernel, dev, seed=4738):
    gen, s = _gen(dev, seed), math.sqrt(n / 1e5)
    parts = []
    for frac, sig, ang in ((0.2, 40, 0.0), (0.3, 60, 2.094), (0.5, 80, 4.189)):
        c = (600 * s * math.cos(ang), 600 * s * math.sin(ang))
        p, vl = _cluster(int(n * frac), c, sig * s, kernel, gen, dev)
        vl = vl - 0.3 * torch.tensor([math.cos(ang), math.sin(ang)], device=dev, dtype=torch.float64) * (n / 1e5) ** 0.25 * 10
        parts.append((p, vl))
    return _finish(torch.cat([o[0] for o in parts]), torch.cat([o[1] for o in parts]))


def plummer(n, kernel, dev, seed=4738):
    gen, s = _gen(dev, seed), math.sqrt(n / 1e5)
    u = torch.rand(n, generator=gen, device=dev, dtype=torch.float64).clamp(max=0.99)
    r = 80 * s * (u / (1 - u)).sqrt()
    th = torch.rand(n, generator=gen, device=dev, dtype=torch.float64) * 2 * math.pi
    pos = torch.stack([r * th.cos(), r * th.sin()], 1)
    return _finish(pos, torch.randn(n, 2, generator=gen, device=dev, dtype=torch.float64) * virial_sigma(pos, kernel))


def disk(n, kernel, dev, seed=4738):
    gen, s = _gen(dev, seed), math.sqrt(n / 1e5)
    rd = 100 * s
    u = torch.rand(n, generator=gen, device=dev, dtype=torch.float64)
    r = torch.empty_like(u)
    grid = torch.linspace(0, 12 * rd, 20000, device=dev, dtype=torch.float64)
    cdf = 1 - (1 + grid / rd) * torch.exp(-grid / rd)
    r = grid[torch.searchsorted(cdf, u).clamp(max=len(grid) - 1)]
    th = torch.rand(n, generator=gen, device=dev, dtype=torch.float64) * 2 * math.pi
    pos = torch.stack([r * th.cos(), r * th.sin()], 1)
    rank = torch.argsort(torch.argsort(r)).double() + 1
    vc = (rank / r.clamp(min=1e-3)).sqrt() * 0.7
    vel = vc[:, None] * torch.stack([-th.sin(), th.cos()], 1) + torch.randn(n, 2, generator=gen, device=dev, dtype=torch.float64) * 0.1 * vc[:, None]
    return _finish(pos, vel)


def clumpy(n, kernel, dev, seed=4738, k=30):
    gen, s = _gen(dev, seed), math.sqrt(n / 1e5)
    nf, per = int(0.3 * n), int(0.7 * n) // k
    cent = torch.randn(k, 2, generator=gen, device=dev, dtype=torch.float64) * 400 * s
    parts = []
    for i in range(k):
        parts.append(_cluster(per, (float(cent[i, 0]), float(cent[i, 1])), 15 * s, kernel, gen, dev))
    fp = torch.randn(nf, 2, generator=gen, device=dev, dtype=torch.float64) * 400 * s
    fv = torch.randn(nf, 2, generator=gen, device=dev, dtype=torch.float64) * 0.5 * (n / 1e5) ** 0.25
    parts.append((fp, fv))
    return _finish(torch.cat([o[0] for o in parts]), torch.cat([o[1] for o in parts]))


IC = {"uniform": uniform, "flyby": flyby, "three": three, "plummer": plummer, "disk": disk, "clumpy": clumpy}
