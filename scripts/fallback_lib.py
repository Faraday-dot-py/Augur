"""Fallback (H3) test harness: corrupted estimator heads, collapse IC, rollout with per-step true-error sampling and full-N audit-noise study."""
import math
import time

import numpy as np
import torch

from scripts import est_train
from scripts import kernels
from scripts import nbody_ic
from scripts import nbody_rollout as nr
from scripts.adaptive_force import AdaptiveForce, sync


class BiasHead:
    def __init__(self, head, b):
        self.head, self.b = head, b

    def __call__(self, x):
        return self.head(x) + self.b


class ConstHead:
    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        return torch.full((x.shape[0],), self.c, dtype=torch.float64, device=x.device)


class NoiseHead:
    """true head + N(0,1). sym=True: noise is a fixed pseudo-random function of the (exchange-symmetric) features, so both orders of a
    pair get the same noise and momentum symmetry is kept. sym=False: fresh noise per ordered pair (generator seed 4738)."""

    def __init__(self, head, sym=True, seed=4738):
        self.head, self.sym = head, sym
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.w = torch.rand(12, generator=self.gen, device="cuda", dtype=torch.float64) * 100

    def __call__(self, x):
        y = self.head(x)
        if self.sym:
            u = torch.sin((x.double() * self.w).sum(1) * 12.9898) * 43758.5453
            u = (u - u.floor()).clamp(1e-6, 1 - 1e-6)
            z = math.sqrt(2) * torch.erfinv(2 * u - 1)
        else:
            z = torch.randn(len(y), generator=self.gen, device=y.device, dtype=torch.float64)
        return y + z


def make_head(path, corrupt, dev):
    h = est_train.load(path, dev)
    if corrupt == "none":
        return h
    kind, _, val = corrupt.partition(":")
    if kind == "bias":
        return BiasHead(h, float(val))
    if kind == "const":
        return ConstHead(float(val))
    if kind == "noise":
        return NoiseHead(h, True)
    if kind == "noise_asym":
        return NoiseHead(h, False)
    raise ValueError(corrupt)


def collapse(n, kernel, dev, seed=4738, radius=150.0, ratio=0.02):
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = radius * torch.rand(n, generator=gen, device=dev, dtype=torch.float64).sqrt()
    th = torch.rand(n, generator=gen, device=dev, dtype=torch.float64) * 2 * math.pi
    pos = torch.stack([r * th.cos(), r * th.sin()], 1)
    sv = nbody_ic.virial_sigma(pos, kernel, ratio)
    vel = torch.randn(n, 2, generator=gen, device=dev, dtype=torch.float64) * sv
    return nbody_ic._finish(pos, vel)


def make_ic(name, n, kernel, dev):
    if name == "collapse":
        return collapse(n, kernel, dev)
    return nbody_ic.IC[name](n, kernel, dev)


def true_err(kernel, pos, a, idx):
    ex = kernel.exact_accel(pos, idx, 128)
    d = a[idx].double() - ex
    return float(d.norm() / ex.norm()), d.pow(2).sum(1), ex.pow(2).sum(1)


def run(cfg, dev=torch.device("cuda")):
    kernel = est_train.make_kernel(cfg["kernel"], dev)
    kernels.current = kernel
    torch.manual_seed(4738)
    n, steps, dt = cfg["n"], cfg["steps"], 0.05
    pos, vel = make_ic(cfg["ic"], n, kernel, dev)
    head = make_head(cfg["est"], cfg["corrupt"], dev) if cfg["mode"] in ("est", "adaptive") else None
    kw = {k: cfg[k] for k in ("target", "audit_every", "audit_k", "lam_max", "reprobe_every", "severe", "fail_limit") if k in cfg}
    af = AdaptiveForce(kernel, head, mode=cfg["mode"], cap=8, tol=1e-3, geo_theta=0.35, device=str(dev), **kw)
    gt = torch.Generator(device=dev).manual_seed(9001)
    target = af.target
    trace, noise, diags = [], [], []
    tk = cfg.get("true_k", 4000)
    ks = cfg.get("noise_ks", [])
    ng = torch.Generator(device=dev).manual_seed(777)

    def force(pos, step):
        probing = af.probing
        t0 = sync()
        a = af(pos)
        rec = af.stats[-1]
        row = {"step": step, "mode": rec["mode"], "probing": bool(probing and rec["mode"] == "est"), "lam": rec["lam"], "theta": rec["theta"],
               "cost": rec["cost"], "time_s": rec["time_s"], "audit": rec["audit"], "audit_time_s": rec.get("audit_time_s"), "true": None}
        if step % cfg.get("true_every", 5) == 0:
            idx = torch.randperm(n, generator=gt, device=dev)[:tk]
            row["true"] = true_err(kernel, pos, a, idx)[0]
        if ks and step % 20 == 0:
            idx = torch.arange(n, device=dev)
            e, d2, m2 = true_err(kernel, pos, a, idx)
            for k in ks:
                au = []
                for _ in range(100):
                    s = torch.randperm(n, generator=ng, device=dev)[:k]
                    au.append(float((d2[s].sum() / m2[s].sum()).sqrt()))
                au = np.array(au)
                noise.append({"step": step, "mode": rec["mode"], "k": k, "true": e, "mean": float(au.mean()), "std": float(au.std()),
                              "p_over": float((au > target).mean()), "p_severe": float((au > af.severe * target).mean())})
        trace.append(row)
        return a

    a = force(pos, 0)
    d0 = nr.diagnostics(pos, vel, kernel, 0)
    d0["sumv"] = float(vel.norm(dim=1).sum())
    diags.append(d0)
    t_run = time.time()
    for step in range(1, steps + 1):
        vel = vel + 0.5 * dt * a
        pos = pos + dt * vel
        a = force(pos, step)
        vel = vel + 0.5 * dt * a
        if step % cfg.get("diag_every", 100) == 0:
            d = nr.diagnostics(pos, vel, kernel, step)
            d["sumv"] = float(vel.norm(dim=1).sum())
            diags.append(d)
            print(f"step {step} E {d['E']:.6g} P/sumv {d['P'] / d['sumv']:.2e} mode {af.now} lam {af.lam:.2f} ev {len(af.events)} t {time.time() - t_run:.0f}s", flush=True)
    return {"cfg": cfg, "wall_s": time.time() - t_run, "trace": trace, "diags": diags, "events": af.events, "noise": noise}
