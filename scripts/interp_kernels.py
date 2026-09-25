"""Kernel family for the interpretability test: power-law softened force f(d) = d (d^2 + eps^2)^-((p+1)/2) (p = 1 is the 2D inverse
distance force, p = 2 the softened gravity of scripts/kernels.analytic), optionally screened by exp(-d/L)."""
import torch

from scripts import kernels


def plw(p, eps=0.5, length=None):
    if length is None:
        return kernels.RadialKernel(lambda d: d * (d ** 2 + eps ** 2) ** (-(p + 1) / 2), f"plw{p:g}")
    return kernels.RadialKernel(lambda d: d * (d ** 2 + eps ** 2) ** (-(p + 1) / 2) * torch.exp(-d / length), f"plw{p:g}_L{length:g}")


def get(name, eps=0.5):
    if name == "analytic":
        return kernels.analytic(eps)
    if name == "inv_distance":
        return kernels.inv_distance(eps)
    if name == "yukawa30":
        return kernels.yukawa(eps, 30.0)
    if name.startswith("plw"):
        return plw(float(name[3:]), eps)
    raise ValueError(name)
