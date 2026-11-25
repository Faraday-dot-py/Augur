"""Cheap far-kernel check (docs/debugging/scatter-tiling.md): learned kmlp(r) vs the true softened force, and a DIAGNOSTIC rerun of the
wide 16-tile scene with the analytic kernel substituted at r > thr (NOT a fix; the analytic kernel is a hard-coded force, used only to
measure how much of the far-term error the kernel explains).
Usage: PYTHONPATH=. python scripts/scatter_far_kernel_check.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/scatter_far_kernel.json
"""
import argparse
import json
import types

import torch

from scripts import scatter_scaling_ops as ops
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling import TiledScatter
from scripts.scatter_tiling_eval import DT, EPS, SUB, load

L, N, SIGMA = 256.0, 4800, 32.0


def k_true(r):
    return -1.0 / torch.sqrt(r ** 2 + EPS ** 2)


def f_true(r):
    return r / (r ** 2 + EPS ** 2) ** 1.5


def split_fac(m, r):
    return 1 - (1 - (r / m.pp).clamp(max=1.0) ** 2) ** 2


def k_learned(m, r):
    return m.kmlp(torch.stack([r, torch.log(r + 0.5)], -1))[..., 0] * split_fac(m, r)


@torch.no_grad()
def phase_kernel(m, dev):
    rows = []
    for r0 in (2, 5, 10, 20, 32, 40, 48, 64, 80, 100, 150, 200, 300, 400, 500):
        r = torch.tensor(float(r0), device=dev, requires_grad=True)
        with torch.enable_grad():
            k = k_learned(m, r)
            dk, = torch.autograd.grad(k, r)
        kt = k_true(r.detach())
        rows.append({"r": r0, "K_learned": float(k), "K_true": float(kt), "F_learned_inward": float(dk), "F_true_inward": float(f_true(r.detach())),
                     "F_ratio": float(dk / f_true(r.detach()))})
        print(rows[-1], flush=True)
    return rows


def hybrid_kfft(thr, mode):
    """mode splice: for r > thr use K_learned(thr) + K_true(r) - K_true(thr) (continuous K, true force beyond thr);
    mode trunc: K constant beyond thr (zero force beyond thr)."""
    def kfft(self, rho):
        G = self.grid
        idx = torch.arange(2 * G, device=rho.device, dtype=rho.dtype)
        d = torch.minimum(idx, 2 * G - idx) * self.h
        r = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        K = k_learned(self, r)
        if thr is not None:
            t = torch.tensor(float(thr), device=rho.device, dtype=rho.dtype)
            k0 = k_learned(self, t)
            tail = k0 + k_true(r) - k_true(t) if mode == "splice" else k0.expand_as(r)
            K = torch.where(r > thr, tail, K)
        return torch.fft.rfft2(K)
    return kfft


def rel(a, ref):
    return float((a - ref).norm(dim=-1).sum() / ref.norm(dim=-1).sum())


@torch.no_grad()
def run_single(m, pos, vel, mass, steps):
    ops.retarget(m, 512, L)
    msk = torch.ones(1, pos.shape[0], device=pos.device)
    st = (m.init_field(1, pos.device)[0], None)
    a0, _ = m.force(pos[None], vel[None], mass[None], msk, st[0])
    p, v = pos[None], vel[None]
    Ps = [p[0]]
    for _ in range(steps):
        p, v, st, _ = m.step(p, v, mass[None], msk, st)
        Ps.append(p[0])
    return a0[0], torch.stack(Ps).double()


@torch.no_grad()
def run_tiled(m, pos, vel, mass, steps, far):
    T = TiledScatter(m, 4, 64, 16, ghosts=True, field_x=True, far=far)
    a0, _ = T.force(pos, vel, mass, T.init_field(pos.device))
    ts = (T.init_field(pos.device), None)
    p, v = pos, vel
    Pl = [p]
    for _ in range(steps):
        p, v, ts = T.step(p, v, mass, ts)
        Pl.append(p)
    return a0, torch.stack(Pl).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    m = load(a.ckpt, dev)
    res = {"kernel": phase_kernel(m, dev), "scene": []}
    orig = m._kernel_fft
    for seed in (4738, 9000, 12000):
        p0, v0 = make_scene(N, SIGMA, 1.0, L, seed, 0)
        Pt, _ = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), 20, DT, SUB, EPS)
        pos = torch.tensor(p0, dtype=torch.float32, device=dev)
        vel = torch.tensor(v0, dtype=torch.float32, device=dev)
        mass = torch.ones(N, device=dev)
        ax = ops.exact_acc(pos.double(), EPS).float()
        row = {"seed": seed}
        for thr, mode in ((None, None), (64.0, "splice"), (32.0, "splice"), (64.0, "trunc"), (32.0, "trunc")):
            tag = "learned" if thr is None else f"{mode}_r>{thr:g}"
            m._kernel_fft = types.MethodType(hybrid_kfft(thr, mode), m)
            a_s, P_s = run_single(m, pos, vel, mass, 20)
            r = {"single": {"force_vs_exact": rel(a_s, ax), "err20": float((P_s[20] - Pt[20]).norm(dim=-1).mean())}}
            for name, far in (("tiled_far", True), ("tiled_nofar", False)):
                a_t, P_t = run_tiled(m, pos, vel, mass, 20, far)
                r[name] = {"force_vs_exact": rel(a_t, ax), "force_vs_single": rel(a_t, a_s),
                           "err20": float((P_t[20] - Pt[20]).norm(dim=-1).mean()), "err5": float((P_t[5] - Pt[5]).norm(dim=-1).mean()),
                           "diff_single20": float((P_t[20] - P_s[20]).norm(dim=-1).mean())}
            row[tag] = r
            print(seed, tag, r, flush=True)
        m._kernel_fft = orig
        res["scene"].append(row)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
