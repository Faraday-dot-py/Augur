"""Retrain only the FFT kernel (everything else frozen) on a much larger r range with exact softened-force targets
(docs/debugging/scatter-far-kernel-fix.md, R1/R2).

Targets: r <= lo  -> the shipped kernel (potential K and force), so the interior path is unchanged;
         r >= hi  -> exact softened force r/(r^2+eps^2)^1.5; between: smoothstep blend of the shipped and exact force.
r is sampled log-uniformly on [1, rmax] (tail represented) plus uniform on [0, 34]. Loss: relative force error for r >= 4, absolute
below, K-matching on r <= lo (anchors the constant), soft hinge against repulsion (F < 0) for r >= 4.
Usage: PYTHONPATH=. python scripts/scatter_far_train_kernel.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --arch mlp --out results/kern_r1.pt
"""
import argparse
import json
import math
import time

import torch

from scripts import scatter_far as sfar
from scripts.scatter_tiling_eval import load


def smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def build_targets(old, r, lo, hi):
    Fo = sfar.force_of(old, r)
    Ko = old(r).detach()
    w = smoothstep((r - lo) / (hi - lo))
    return Ko, Fo, (1 - w) * Fo + w * sfar.f_true(r)


def forces(mod, r):
    r = r.clone().requires_grad_(True)
    K = mod(r)
    F, = torch.autograd.grad(K.sum(), r, create_graph=True)
    return K, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rmax", type=float, default=640.0)
    ap.add_argument("--lo", type=float, default=28.0)
    ap.add_argument("--hi", type=float, default=48.0)
    ap.add_argument("--w-k", type=float, default=100.0)
    ap.add_argument("--w-hinge", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=4738)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    m = load(a.ckpt, dev)
    old = sfar.old_fn(m)
    mod = sfar.ARCHS[a.arch](pp=m.pp, split=m.split).to(dev)
    if a.arch == "mlp":
        mod.net.load_state_dict(m.kmlp.state_dict())
    opt = torch.optim.Adam(mod.parameters(), lr=a.lr)
    nA, nB = 2048, 4096
    lrmax = math.log10(a.rmax)
    t0 = time.time()
    best = (1e9, None)
    val_r = torch.logspace(0, lrmax, 6000, device=dev)
    for it in range(a.steps + 1):
        lr = a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        rA = torch.rand(nA, device=dev) * 34.0
        rB = 10 ** (torch.rand(nB, device=dev) * lrmax)
        r = torch.cat([rA, rB]).clamp_min(1e-3)
        Ko, Fo, Ft = build_targets(old, r, a.lo, a.hi)
        K, F = forces(mod, r)
        far = r >= 4
        relerr = ((F - Ft) / Ft.clamp_min(1e-12)).clamp(-5, 5)
        l_rel = (relerr[far] ** 2).mean()
        l_abs = ((F - Ft)[~far] ** 2).mean()
        near = r <= a.lo
        l_k = ((K - Ko)[near] ** 2).mean()
        l_hinge = torch.relu(-F[far] / Ft[far].clamp_min(1e-12)).mean()
        loss = l_rel + l_abs + a.w_k * l_k + a.w_hinge * l_hinge
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 1000 == 0:
            with torch.enable_grad():
                _, Fv = forces(mod, val_r)
            ratio = Fv.detach() / sfar.f_true(val_r)
            sel = (val_r >= 4) & (val_r <= 512)
            dev_max = float((ratio[sel] - 1).abs().max())
            Fo_v = sfar.force_of(old, val_r)
            s32 = (val_r >= 1) & (val_r <= 32)
            near_dev = float(((Fv.detach() - Fo_v).abs() / Fo_v.abs().clamp_min(1e-3))[s32].max())
            score = dev_max + 5 * max(0.0, near_dev - 0.02)
            print(f"it {it} loss {float(loss):.3e} rel {float(l_rel):.3e} k {float(l_k):.3e} max|ratio-1| [4,512] {dev_max:.3f} near dev vs old {near_dev:.4f} "
                  f"minF {float(Fv.min()):.2e} t {time.time() - t0:.0f}s", flush=True)
            if score < best[0]:
                best = (score, {k: v.detach().clone() for k, v in mod.state_dict().items()}, it)
    mod.load_state_dict(best[1])
    print("best it", best[2], "score", best[0], flush=True)
    torch.save({"arch": a.arch, "kw": {"pp": m.pp, "split": m.split}, "state": best[1],
                "meta": {"args": vars(a), "best_it": best[2], "score": best[0]}}, a.out)
    json.dump({"args": vars(a), "best_it": best[2], "score": best[0]}, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
