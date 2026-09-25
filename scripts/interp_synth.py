"""Controlled synthetic node pairs for the interpretability test: point-cloud nodes, the exact node-pair label (same definition as
dual_estimator.label: rms over target particles of |exact sum - first-order local expansion|, over m_B |g|, max of both orders), the same
12 features as dual_estimator.pair_feats, and the second-order Taylor theory of the label."""
from types import SimpleNamespace

import torch

from scripts import dual_estimator as de

EPS = de.EPS


def rot(t):
    c, s = torch.cos(t), torch.sin(t)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def square_cloud(P, n, size, gen, dev):
    """uniform points in a square cell of side `size` (P,), random orientation: (P,n,2)."""
    pts = (torch.rand(P, n, 2, device=dev, generator=gen, dtype=torch.float64) - 0.5) * size[:, None, None]
    return pts @ rot(torch.rand(P, device=dev, generator=gen, dtype=torch.float64) * 6.2831853).transpose(1, 2)


def ring_pattern(n_rings=4, m=16, dev="cpu"):
    """n_rings*m points: equally spaced points on equal-area rings filling the unit disk (isotropic to high multipole order)."""
    pts = []
    for k in range(n_rings):
        r = ((k + 0.5) / n_rings) ** 0.5
        ph = 2 * torch.pi * torch.arange(m, dtype=torch.float64) / m + 0.37 * k
        pts.append(torch.stack([r * torch.cos(ph), r * torch.sin(ph)], 1))
    return torch.cat(pts).to(dev)


def ellipse_cloud(P, R, kappa, along, gen, dev, tiny=1.0):
    """ring pattern mapped affinely to an ellipse of area pi R^2 and trace fixed as kappa varies (semi-axes a, b=a/kappa,
    a^2 + b^2 = 2 R^2), major axis along the +x axis (along=True) or along y. R: scalar or (P,)."""
    base = ring_pattern(dev=dev)
    a = (2 * kappa ** 2 / (1 + kappa ** 2)) ** 0.5
    b = a / kappa
    ax = torch.tensor([a, b] if along else [b, a], dtype=torch.float64, device=dev)
    R = torch.as_tensor(R, dtype=torch.float64, device=dev).reshape(-1, 1, 1) * tiny
    return base[None] * ax * R


def feats(cA, cB, sizeA, sizeB, count):
    """(P,n,2) clouds, cell sizes (P,), count -> (P,12) features as in dual_estimator.pair_feats plus the label geometry."""
    P = cA.shape[0]
    comA, comB = cA.mean(1), cB.mean(1)

    def q(c, com):
        dv = c - com[:, None]
        return torch.stack([(dv[..., 0] ** 2).mean(1), (dv[..., 0] * dv[..., 1]).mean(1), (dv[..., 1] ** 2).mean(1)], 1)
    fl = SimpleNamespace(com=torch.cat([comA, comB]), q=torch.cat([q(cA, comA), q(cB, comB)]), size=torch.cat([sizeA, sizeB]),
                         count=torch.full((2 * P,), count, dtype=torch.long, device=cA.device))
    ga = torch.arange(P, device=cA.device)
    return de.pair_feats(fl, ga, ga + P)


@torch.no_grad()
def one_side(kernel, cA, cB, chunk=500):
    """rms over targets in A of |sum_j f_ij - (n_B g(r) - n_B G (x_i - com_A))| and |g(r)|."""
    P, n = cA.shape[:2]
    comA, comB = cA.mean(1), cB.mean(1)
    r = comB - comA
    g, grad = kernel.g_grad(r)
    e2 = torch.empty(P, dtype=torch.float64, device=cA.device)
    for i in range(0, P, chunk):
        a, b = cA[i:i + chunk], cB[i:i + chunk]
        d = b[:, None] - a[:, :, None]
        fs = kernel.pair(d.reshape(-1, 2)).view(d.shape[0], n, cB.shape[1], 2).sum(2)
        dx = a - comA[i:i + chunk, None]
        gg = grad[i:i + chunk, None] * cB.shape[1]
        ap = g[i:i + chunk, None] * cB.shape[1] - torch.stack([gg[..., 0] * dx[..., 0] + gg[..., 1] * dx[..., 1],
                                                              gg[..., 2] * dx[..., 0] + gg[..., 3] * dx[..., 1]], -1)
        e2[i:i + chunk] = ((fs - ap) ** 2).sum(-1).mean(1)
    return e2.sqrt(), g.norm(dim=1)


def label(kernel, cA, cB):
    """log of max(e_A, e_B) / (m |g|) with equal masses."""
    ea, gm = one_side(kernel, cA, cB)
    eb, _ = one_side(kernel, cB, cA)
    return torch.log((torch.maximum(ea, eb) / (cA.shape[1] * gm)).clamp(min=1e-12))


def hessian(kernel, r):
    """H[k,a,b] = d^2 g_k / dx_a dx_b of the force field g(x) = f(|x|) x/|x| at r (P,2): (P,2,2,2)."""
    from torch.func import jacfwd, vmap

    def g(x):
        n = x.norm()
        return kernel.f(n) * x / n
    return vmap(jacfwd(jacfwd(g)))(r)


def taylor_side(kernel, cA, cB):
    """second-order Taylor prediction of one_side / (n_B |g|): residual_i = 1/2 H:(Q_B + u_i u_i), u_i = x_i - com_A."""
    comA, comB = cA.mean(1), cB.mean(1)
    r = comB - comA
    H = hessian(kernel, r)
    dv = cB - comB[:, None]
    QB = torch.einsum("pna,pnb->pab", dv, dv) / cB.shape[1]
    u = cA - comA[:, None]
    M = QB[:, None] + torch.einsum("pna,pnb->pnab", u, u)
    res = 0.5 * torch.einsum("pkab,pnab->pnk", H, M)
    gm = kernel.pair(r).norm(dim=1)
    return res.pow(2).sum(-1).mean(1).sqrt() / gm


def taylor_label(kernel, cA, cB):
    return torch.log(torch.maximum(taylor_side(kernel, cA, cB), taylor_side(kernel, cB, cA)).clamp(min=1e-12))
