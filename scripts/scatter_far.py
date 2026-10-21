"""Opt-in replacement kernels for the scatter-field model's FFT kernel `kmlp(r)` (docs/debugging/scatter-far-kernel-fix.md).

Nothing in scatter_field.py / scatter_tiling.py is edited: `install(model, kfun)` swaps `model._kernel_fft`, so the near grid, the
tiled copies and the coarse far term (TiledScatter._phi) all use `kfun(r)` = potential kernel K(r) (acceleration = +dK/dr inward).

Kernel specs (strings, used by the eval/training scripts):
  old                  untouched model (second MLP feature log(r + h) with h = the current grid's cell, as shipped)
  oldf                 old kmlp but feature fixed to log(r + 0.5) at every level (what the new kernels use)
  clamp<rc>_zero       DIAGNOSTIC REFERENCE: old kmlp(min(r, rc)), zero force beyond rc (hard clamp)
  clamp<rc>_inv2       DIAGNOSTIC REFERENCE: old kmlp up to rc, then F(rc) (rc/r)^2 (hard 1/r^2 continuation)
  oracle<thr>          DIAGNOSTIC ONLY: analytic softened force beyond thr (hard-coded force)
  file:<path>          learned replacement kernel saved by scatter_far_train_kernel.py
"""
import types

import torch
import torch.nn as nn

EPS = 0.5


def k_true(r):
    return -1.0 / torch.sqrt(r ** 2 + EPS ** 2)


def f_true(r):
    return r / (r ** 2 + EPS ** 2) ** 1.5


def split_fac(m, r):
    if not m.split:
        return torch.ones_like(r)
    return 1 - (1 - (r / m.pp).clamp(max=1.0) ** 2) ** 2


def feat(r):
    return torch.stack([r, torch.log(r + 0.5)], -1)


def old_fn(m, fixed_feat=True):
    def k(r):
        return m.kmlp(feat(r))[..., 0] * split_fac(m, r)
    return k


def force_of(kfun, r):
    """inward force dK/dr at r (1-D tensor) by autograd."""
    r = r.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        k = kfun(r)
        f, = torch.autograd.grad(k.sum(), r)
    return f.detach()


def clamp_fn(m, rc, mode):
    base = old_fn(m)
    rct = torch.tensor(float(rc))
    state = {}

    def k(r):
        dev = r.device
        if "f" not in state or state["dev"] != dev:
            state["f"] = float(force_of(base, rct.to(dev)[None])[0])
            state["dev"] = dev
        K = base(r.clamp(max=rc))
        if mode == "inv2":
            tail = state["f"] * rc ** 2 * (1.0 / rc - 1.0 / r.clamp(min=rc))
            K = K + tail
        return K
    return k


def oracle_fn(m, thr):
    base = old_fn(m)

    def k(r):
        t = torch.tensor(float(thr), device=r.device, dtype=r.dtype)
        k0 = base(t)
        tail = k0 + k_true(r) - k_true(t)
        return torch.where(r > thr, tail, base(r))
    return k


class MLPKernel(nn.Module):
    """Same architecture as kmlp (2 -> 64 -> 64 -> 1, GELU), feature (r, log(r + 0.5)), times the model's pp split factor."""

    def __init__(self, pp=2.0, split=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.pp, self.split = pp, split

    def forward(self, r):
        K = self.net(feat(r))[..., 0]
        if self.split:
            K = K * (1 - (1 - (r / self.pp).clamp(max=1.0) ** 2) ** 2)
        return K


def load_kernel(path, dev):
    st = torch.load(path, map_location=dev, weights_only=False)
    arch = st.get("arch", "mlp")
    mod = ARCHS[arch](**st.get("kw", {})).to(dev)
    mod.load_state_dict(st["state"])
    return mod.eval()


ARCHS = {"mlp": MLPKernel}


def kfun_from_spec(m, spec):
    if spec == "old":
        return None
    if spec == "oldf":
        return old_fn(m)
    if spec.startswith("clamp"):
        rc, mode = spec[5:].split("_")
        return clamp_fn(m, float(rc), mode)
    if spec.startswith("oracle"):
        return oracle_fn(m, float(spec[6:]))
    if spec.startswith("file:"):
        mod = load_kernel(spec[5:], next(m.parameters()).device)
        return mod
    raise ValueError(spec)


def install(model, kfun):
    """Swap the model's FFT kernel for K = kfun(r). kfun None restores the shipped one."""
    if not hasattr(model, "_kernel_fft_orig"):
        model._kernel_fft_orig = model._kernel_fft
    if kfun is None:
        model._kernel_fft = model._kernel_fft_orig
        return model

    def kfft(self, rho):
        G = self.grid
        idx = torch.arange(2 * G, device=rho.device, dtype=rho.dtype)
        d = torch.minimum(idx, 2 * G - idx) * self.h
        r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        K = kfun(r)
        return torch.fft.rfft2(K)
    model._kernel_fft = types.MethodType(kfft, model)
    return model


def install_spec(model, spec):
    install(model, kfun_from_spec(model, spec))
    return model


@torch.no_grad()
def ratio_table(kfun, dev, rmax=640.0, n=4000):
    """force ratio F_kernel / F_true on a log grid in [1, rmax] and summary checks."""
    r = torch.logspace(0, torch.log10(torch.tensor(rmax)).item(), n, device=dev)
    F = force_of(kfun, r)
    ratio = F / f_true(r)
    return r, F, ratio
