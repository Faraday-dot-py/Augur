"""Protoplanetary-disc initial conditions, all particles equal mass: a compact Plummer-like core (the "central body", ids 0..ncore-1) + a disc on near-circular orbits with Toomre-Q velocity dispersion and seeded azimuthal density modes. Softened 2D gravity (G=1, eps .5), same units as the scatter-field models.

Planets are not placed: any orbiting body must form by collective clumping of the disc (the mesh model has no merging). A single heavy star particle was tried first (pilot1, jobs 3733) and destroys the disc in the mesh model, so the central mass is a particle cloud."""
import math

import torch

EPS = 0.5
SEED_R = (10.0, 6.2, 3.8, 16.0)
SEED_M = (0.1, 0.1, 0.1, 0.1)

PRESETS = {
    "d1": dict(mstar=30.0, mdisc=12.0, rin=3.0, rout=14.0, p=1.0, Q=0.9, amp=0.04, mmin=2, mmax=8, acore=1.0, ctrunc=3.0, cdisp=0.6, nseed=0),
}


def make_disc(n, P, seed=4738, device="cuda", dtype=torch.float64):
    """Returns pos, vel, mass of n + nseed particles (core first, then disc, then the nseed placed protoplanets of mass SEED_M on circular orbits at SEED_R), ncore. Centre of mass at rest at the origin."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    mtot = P["mstar"] + P["mdisc"]
    ns = P["nseed"]
    ncore = int(round(n * P["mstar"] / mtot))
    nd = n - ncore
    mp = mtot / n
    kw = dict(device=device, dtype=dtype, generator=g)
    a_c = P["acore"]
    umax = P["ctrunc"] ** 2 / (P["ctrunc"] ** 2 + 1)
    u = torch.rand(ncore, **kw) * umax
    rc = a_c * (u / (1 - u)).sqrt()
    thc = torch.rand(ncore, **kw) * 2 * math.pi
    menc_c = P["mstar"] * (rc ** 2 / (rc ** 2 + a_c ** 2)) / umax
    vcc = (menc_c * rc ** 2 / (rc ** 2 + EPS ** 2) ** 1.5).sqrt()
    sc = P["cdisp"] * vcc
    cpos = rc[:, None] * torch.stack([torch.cos(thc), torch.sin(thc)], 1)
    cvel = torch.randn(ncore, 2, **kw) * sc[:, None]
    p, rin, rout = P["p"], P["rin"], P["rout"]
    a = 2.0 - p
    u = torch.rand(nd, **kw)
    r = (rin ** a + u * (rout ** a - rin ** a)) ** (1.0 / a)
    modes = torch.arange(P["mmin"], P["mmax"] + 1, device=device, dtype=dtype)
    phase = torch.rand(len(modes), **kw) * 2 * math.pi
    th = torch.empty(nd, device=device, dtype=dtype)
    todo = torch.arange(nd, device=device)
    bound = 1 + P["amp"] * len(modes)
    while todo.numel():
        t = torch.rand(todo.numel(), **kw) * 2 * math.pi
        f = 1 + P["amp"] * torch.cos(modes[None] * t[:, None] + phase[None]).sum(1)
        ok = torch.rand(todo.numel(), **kw) * bound < f
        th[todo[ok]] = t[ok]
        todo = todo[~ok]
    sig0 = P["mdisc"] * a / (2 * math.pi * (rout ** a - rin ** a))
    sigma = sig0 * r ** -p
    menc = P["mstar"] + P["mdisc"] * (r ** a - rin ** a) / (rout ** a - rin ** a)
    om = (menc / (r ** 2 + EPS ** 2) ** 1.5).sqrt()
    vc = om * r
    sr = P["Q"] * 3.36 * sigma / om
    sp = sr * 0.5
    er = torch.stack([torch.cos(th), torch.sin(th)], 1)
    et = torch.stack([-torch.sin(th), torch.cos(th)], 1)
    zr = torch.randn(nd, **kw)
    zp = torch.randn(nd, **kw)
    pos = torch.cat([cpos, r[:, None] * er])
    vel = torch.cat([cvel, (sr * zr)[:, None] * er + (vc + sp * zp)[:, None] * et])
    mass = torch.full((n,), mp, device=device, dtype=dtype)
    if ns:
        sr_ = torch.tensor(SEED_R[:ns], device=device, dtype=dtype)
        sm = torch.tensor(SEED_M[:ns], device=device, dtype=dtype)
        sth = torch.rand(ns, **kw) * 2 * math.pi
        smenc = P["mstar"] + P["mdisc"] * ((sr_ ** a - rin ** a) / (rout ** a - rin ** a)).clamp(0, 1)
        svc = (smenc * sr_ ** 2 / (sr_ ** 2 + EPS ** 2) ** 1.5).sqrt()
        pos = torch.cat([pos, sr_[:, None] * torch.stack([torch.cos(sth), torch.sin(sth)], 1)])
        vel = torch.cat([vel, svc[:, None] * torch.stack([-torch.sin(sth), torch.cos(sth)], 1)])
        mass = torch.cat([mass, sm])
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    pos -= (mass[:, None] * pos).sum(0) / mass.sum()
    return pos, vel, mass, ncore
