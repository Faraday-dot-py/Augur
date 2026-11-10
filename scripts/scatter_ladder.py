"""Gaussian-ladder far kernel for ScatterField (docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md, experiment 6).

The FFT far kernel K(r) is replaced by sum_l a_l exp(-r^2 / 2 s_l^2), s_l = S0 * 2^l, each term applied as a normalised Gaussian blur on a grid
coarsened to its width (separable stencil of +-TAPS coarse-cell widths), then bilinearly upsampled. No FFT; the amplitudes a_l are free in sign.
Here they are fitted to a frozen learned kernel by weighted least squares; they are not read from a force law. Cost per level is linear in
the cell count of its coarsened grid, so the whole ladder is linear in grid cells (about 1.3x the finest-level cost).
"""
import math

import torch
import torch.nn.functional as F

from scripts import scatter_field as sf

LEVELS, S0, TAPS = 7, 0.5, 4


def sigmas(levels=LEVELS, s0=S0):
    return [s0 * 2 ** l for l in range(levels)]


def window(model, r):
    return 1 - (1 - (r / model.ppe()).clamp(max=1.0) ** 2) ** 2


@torch.no_grad()
def learned_kernel(model, G=None):
    """Windowed learned kernel on the same 2G x 2G folded lattice that _kernel_fft transforms, returned as (r, K)."""
    G = G or model.grid
    idx = torch.arange(2 * G, dtype=torch.float64)
    d = torch.minimum(idx, 2 * G - idx) * model.h
    r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
    K = model.kmlp(model._feat(r.float(), "k"))[..., 0].double() * window(model, r)
    return r, K


def fit_amps(r, K, sig):
    """Weighted least squares for the amplitudes with weights 1 / (|K| + 0.01 max|K|), i.e. roughly relative error; one fit, no free search."""
    sel = (r <= r.max() / math.sqrt(2) + 1e-9)
    rr, kk = r[sel], K[sel]
    assert kk.abs().max() > 0, "kernel is identically zero"
    A = torch.stack([torch.exp(-rr ** 2 / (2 * s ** 2)) for s in sig], -1)
    w = 1.0 / (kk.abs() + 0.01 * kk.abs().max())
    return torch.linalg.lstsq(A * w[:, None], (kk * w)[:, None]).solution[:, 0]


def fit_error(r, K, amps, sig):
    """Same selection and weights as fit_amps: weighted mean absolute error divided by the weighted mean |K|."""
    sel = (r <= r.max() / math.sqrt(2) + 1e-9)
    rr, kk = r[sel], K[sel]
    w = 1.0 / (kk.abs() + 0.01 * kk.abs().max())
    return float(((ladder_kernel(amps, sig, rr) - kk).abs() * w).sum() / (kk.abs() * w).sum())


def ladder_kernel(amps, sig, r):
    return sum(a * torch.exp(-r ** 2 / (2 * s ** 2)) for a, s in zip(amps, sig))


def read_exponent(amps, sig, lo=2):
    """Slope of log|a_l| against log s_l over levels lo.. gives the potential exponent p (K ~ r^-p); the force exponent is p + 1."""
    x = torch.log(torch.as_tensor(sig[lo:], dtype=torch.float64))
    y = torch.log(amps[lo:].abs())
    xm, ym = x.mean(), y.mean()
    slope = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    return float(-slope) + 1.0


def _taps(sc, dtype, device):
    R = int(TAPS * sc)
    k = torch.exp(-torch.arange(-R, R + 1, device=device, dtype=dtype) ** 2 / (2 * sc ** 2))
    return k / k.sum(), R


def _fine_sum(s_cells):
    R = int(6 * s_cells)
    k = torch.exp(-torch.arange(-R, R + 1, dtype=torch.float64) ** 2 / (2 * s_cells ** 2)).sum()
    return float(k * k)


def level_blur(rho, l):
    """Normalised Gaussian blur of width s_l (fine cells 2^l) on the grid coarsened by 2^max(l - 1, 0); the stencil width is 1 or 2 coarse cells."""
    f = 2 ** max(l - 1, 0)
    sc = 2.0 ** l / f
    x = F.avg_pool2d(rho, f) if f > 1 else rho
    k, R = _taps(sc, rho.dtype, rho.device)
    e = 1 if f > 1 else 0
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, R + e))
    x = F.conv2d(x, k.view(1, 1, -1, 1), padding=(R + e, 0))
    if f == 1:
        return x
    x = F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)
    return x[..., f:-f, f:-f]


def ladder_potential(rho, amps, h, levels=None):
    """phi = h^2 sum_j K(x - x_j) rho_j with K the ladder; rho is the (B,1,H,W) density channel of ScatterField.scatter."""
    out = 0
    for l, a in enumerate(amps):
        if levels is not None and l not in levels:
            continue
        s_cells = (S0 * 2 ** l) / h
        out = out + float(a) * h * h * _fine_sum(s_cells) * level_blur(rho, l)
    return out


def ladder_potential_tiled(rho, amps, h, tile, halo):
    """Single-grid result recomputed tile by tile from only the density inside tile + halo (cells), zero outside; no global term."""
    B, _, H, W = rho.shape
    out = torch.zeros_like(rho)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            ya, yb = y0 - halo, y0 + tile + halo
            xa, xb = x0 - halo, x0 + tile + halo
            win = torch.zeros(B, 1, tile + 2 * halo, tile + 2 * halo, dtype=rho.dtype, device=rho.device)
            sy0, sy1, sx0, sx1 = max(ya, 0), min(yb, H), max(xa, 0), min(xb, W)
            win[:, :, sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa] = rho[:, :, sy0:sy1, sx0:sx1]
            p = ladder_potential(win, amps, h)
            out[:, :, y0:y0 + tile, x0:x0 + tile] = p[:, :, halo:halo + tile, halo:halo + tile]
    return out


class LadderScatterField(sf.ScatterField):
    """ScatterField with the FFT kernel path replaced by the ladder; kmlp is unused. Set model.amps before use."""

    def set_amps(self, amps):
        self.amps = [float(a) for a in amps]

    def kernel_acc(self, rho):
        return self.neg_grad(ladder_potential(rho, self.amps, self.h))
