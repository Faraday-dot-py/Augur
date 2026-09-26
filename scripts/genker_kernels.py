"""Extra radial kernels for the force-law generalisation test (genker): softened gravity eps 2, softened d^-1.5 power law, and a
Lennard-Jones-like smooth radial force with a repulsive core (sign change at d = 2). All plug into kernels.RadialKernel (f > 0 attracts).
"""
import torch

from scripts import kernels
from scripts.kernels import RadialKernel

LJ_EPS, LJ_R0 = 1.0, 2.0


def softgrav2():
    return RadialKernel(lambda d: d * (d ** 2 + 4.0) ** -1.5, "softgrav2")


def pow15(eps=0.5):
    return RadialKernel(lambda d: d * (d ** 2 + eps ** 2) ** -1.25, "pow15")


def lj():
    def f(d):
        s = d ** 2 + LJ_EPS ** 2
        return d * s ** -1.5 * (1 - (LJ_R0 ** 2 / s) ** 2)
    return RadialKernel(f, "lj")


NAMES = ["analytic", "learned", "inv_distance", "yukawa30", "softgrav2", "pow15", "lj"]


def make_kernel(name, dev, ckpt="checkpoints/gravity_central_v1.pt"):
    if name == "learned":
        k = kernels.learned(ckpt, dev)
        f1 = k.f
        return RadialKernel(lambda d: f1(d.reshape(-1)).reshape(d.shape), "learned")
    return {"analytic": kernels.analytic, "inv_distance": kernels.inv_distance, "yukawa30": kernels.yukawa, "yukawa": kernels.yukawa,
            "softgrav2": softgrav2, "pow15": pow15, "lj": lj}[name]()
