"""Tier >= 1 interface from docs/theory/heterogeneous-tiered.md ("Tier k>=1 interface" correction): each coarse cell c carries a polynomial potential
phi_c(x) = phi0 + g.d + 0.5 d^T H d (d = x - cell centre, H symmetric), blended across the 4 nearest cells with bilinear partition-of-unity weights;
force = -grad of the blended potential (autograd, so the grad-of-weight term is included). `naive_force` blends per-cell forces -(g + H d) without that
term (the form the doc says breaks curl-free). Coefficient layout (B, 6, G, G): phi0, gx, gy, hxx, hxy, hyy.
"""
import torch


def _corners(pos, G, h, extent):
    u = (pos + extent / 2) / h - 0.5
    i0 = torch.floor(u)
    return i0.long(), u - i0.detach()


def _cells(coef, pos, h, extent):
    """Yield (weight, phi_c(x), -(g + H d)) for the 4 corner cells of every position. pos (B, N, 2)."""
    B, C, G, _ = coef.shape
    i0, w = _corners(pos.detach(), G, h, extent)
    u = (pos + extent / 2) / h - 0.5
    w = u - i0.to(pos.dtype)
    flat = coef.permute(0, 2, 3, 1).reshape(B, G * G, C)
    for ox in (0, 1):
        for oy in (0, 1):
            ix, iy = i0[..., 0] + ox, i0[..., 1] + oy
            ok = ((ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)).to(pos.dtype)
            idx = iy.clamp(0, G - 1) * G + ix.clamp(0, G - 1)
            c = torch.gather(flat, 1, idx[..., None].expand(-1, -1, C))
            ctr = torch.stack([(ix.to(pos.dtype) + 0.5) * h - extent / 2, (iy.to(pos.dtype) + 0.5) * h - extent / 2], -1)
            d = pos - ctr
            wt = (w[..., 0] if ox else 1 - w[..., 0]) * (w[..., 1] if oy else 1 - w[..., 1]) * ok
            gx, gy, hxx, hxy, hyy = c[..., 1], c[..., 2], c[..., 3], c[..., 4], c[..., 5]
            phi = c[..., 0] + gx * d[..., 0] + gy * d[..., 1] + 0.5 * (hxx * d[..., 0] ** 2 + 2 * hxy * d[..., 0] * d[..., 1] + hyy * d[..., 1] ** 2)
            f = torch.stack([-(gx + hxx * d[..., 0] + hxy * d[..., 1]), -(gy + hxy * d[..., 0] + hyy * d[..., 1])], -1)
            yield wt, phi, f


def potential(coef, pos, h, extent):
    return sum(wt * phi for wt, phi, _ in _cells(coef, pos, h, extent))


def force(coef, pos, h, extent, create_graph=False):
    """Exact gradient force -grad(blended potential) at pos (B, N, 2)."""
    with torch.enable_grad():
        p = pos if pos.requires_grad else pos.detach().requires_grad_(True)
        U = potential(coef, p, h, extent).sum()
        return -torch.autograd.grad(U, p, create_graph=create_graph)[0]


def naive_force(coef, pos, h, extent):
    """Blend of per-cell forces without the grad-of-weight term (not a gradient in general)."""
    return sum(wt[..., None] * f for wt, _, f in _cells(coef, pos, h, extent))


def loop_work(fn, centre, radius, h, extent, n=400):
    """Line integral of force around a circle, per unit circumference-radius (zero for a conservative force). centre (2,) tensor."""
    t = torch.linspace(0, 2 * torch.pi, n + 1, dtype=centre.dtype, device=centre.device)[:-1]
    pts = centre + radius * torch.stack([torch.cos(t), torch.sin(t)], -1)
    tang = torch.stack([-torch.sin(t), torch.cos(t)], -1)
    f = fn(pts[None])[0]
    return float((f * tang).sum() * (2 * torch.pi * radius / n))
