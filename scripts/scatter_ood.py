"""Zero-shot OOD stress of the scatter-field Exp E checkpoint, one axis at a time, vs an exact float64 softened all-pairs reference.

Scenes are generated centred at the grid origin. Truth: kick-drift-kick, h = 0.025 per substep (4 substeps at dt 0.1, as in training), float64, optional
relativistic momentum state (c). Model: Newtonian scenes use ScatterField.step (Verlet, model.dt = tick dt); relativistic scenes use the scatter_bh.py KDK on
momentum. Errors are mean per-body position error vs truth at physical times 0.5/1/2/5 (ticks 5/10/20/50 at dt 0.1; the last informative BH horizon per
chaos-floor-investigation.md). Floor = truth with a 1e-5 * rms-radius position perturbation. Ballistic = constant-velocity coasting error.

Usage: PYTHONPATH=. python scripts/scatter_ood.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/scatter_ood.json [--axes a,b] [--traj results/scatter_ood_traj.npz]
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

SEEDS = (9100, 9200, 9300, 4738)
TIMES = (0.5, 1.0, 2.0, 5.0)
T_TOTAL = 10.0
H = 0.025
EXTENT = 64.0
BIG_C = None


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def mom_from_vel(vel, c):
    if c is None:
        return vel
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    return vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()


def truth(pos0, vel0, mass, steps, dt, eps, c, dev):
    sub = max(1, int(round(dt / H)))
    h = dt / sub
    pos = torch.tensor(pos0, dtype=torch.float64, device=dev)
    vel = torch.tensor(vel0, dtype=torch.float64, device=dev)
    m = torch.tensor(mass, dtype=torch.float64, device=dev)

    def acc(p):
        d = p[None] - p[:, None]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
        inv.fill_diagonal_(0.0)
        return (d * inv[..., None] * m[None, :, None]).sum(1)

    mom = mom_from_vel(vel, c)
    sp = (lambda p: p) if c is None else (lambda p: speed(p, c))
    ps, vs = [pos.clone()], [sp(mom).clone()]
    a = acc(pos)
    for _ in range(steps):
        for _ in range(sub):
            mom = mom + 0.5 * h * a
            pos = pos + h * sp(mom)
            a = acc(pos)
            mom = mom + 0.5 * h * a
        ps.append(pos.clone())
        vs.append(sp(mom).clone())
    return torch.stack(ps), torch.stack(vs)


def model_rollout(model, pos0, vel0, mass, steps, dt, c, dev, dtype=torch.float32):
    n = pos0.shape[0]
    pos = torch.tensor(pos0, dtype=dtype, device=dev)[None]
    vel = torch.tensor(vel0, dtype=dtype, device=dev)[None]
    m = torch.tensor(mass, dtype=dtype, device=dev)[None]
    mask = torch.ones(1, n, dtype=dtype, device=dev)
    model.dt = dt
    ps, vs = [pos[0].clone()], [vel[0].clone()]
    field = model.init_field(1, dev, dtype)
    nonfinite = None
    with torch.no_grad():
        if c is None:
            for k in range(steps):
                pos, vel, field, _ = model.step(pos, vel, m, mask, field)
                ps.append(pos[0].clone())
                vs.append(vel[0].clone())
                if nonfinite is None and not (torch.isfinite(pos).all() and torch.isfinite(vel).all()):
                    nonfinite = k + 1
                    break
        else:
            mom = mom_from_vel(vel, c)
            field = model.init_field(1, dev, dtype)[0]
            a, field = model.force(pos, speed(mom, c), m, mask, field)
            for k in range(steps):
                mom = mom + 0.5 * dt * a
                pos = pos + dt * speed(mom, c)
                a, field = model.force(pos, speed(mom, c), m, mask, field)
                mom = mom + 0.5 * dt * a
                ps.append(pos[0].clone())
                vs.append(speed(mom, c)[0].clone())
                if nonfinite is None and not (torch.isfinite(pos).all() and torch.isfinite(mom).all()):
                    nonfinite = k + 1
                    break
    P, V = torch.stack(ps).double(), torch.stack(vs).double()
    return P, V, nonfinite


def energy_t(P, V, m, eps, c):
    """(T,) total energy: sum m c^2 (gamma-1) or 0.5 m v^2, minus sum_{i<j} m_i m_j / softened r."""
    if c is None:
        ke = 0.5 * (m * (V ** 2).sum(-1)).sum(-1)
    else:
        v2 = (V ** 2).sum(-1).clamp(max=0.9801 * c ** 2)
        ke = (m * c ** 2 * ((1 - v2 / c ** 2).rsqrt() - 1)).sum(-1)
    d = P[:, None] - P[:, :, None]
    r = ((d ** 2).sum(-1) + eps ** 2).sqrt()
    n = P.shape[1]
    iu = torch.triu_indices(n, n, 1, device=P.device)
    pe = -(m[iu[0]] * m[iu[1]] * (1.0 / r[:, iu[0], iu[1]])).sum(-1)
    return ke + pe, ke, pe


def mom_t(V, m, c):
    if c is None:
        p = V
    else:
        v2 = (V ** 2).sum(-1, keepdim=True).clamp(max=0.9801 * c ** 2)
        p = V * (1 - v2 / c ** 2).rsqrt()
    return p


def conserved(P, V, m, eps, c):
    E, ke, pe = energy_t(P, V, m, eps, c)
    scale = (ke[0].abs() + pe[0].abs()).clamp_min(1e-9)
    dE = ((E - E[0]).abs() / scale)
    p = mom_t(V, m, c)
    Ptot = (m[None, :, None] * p).sum(1)
    pscale = (m[None, :, None] * p.norm(dim=-1, keepdim=True)).sum(1)[0].clamp_min(1e-9)
    dP = (Ptot - Ptot[0]).norm(dim=-1) / pscale
    com = (m[None, :, None] * P).sum(1, keepdim=True) / m.sum()
    r = P - com
    L = (m * (r[..., 0] * p[..., 1] - r[..., 1] * p[..., 0])).sum(-1)
    lscale = (m * r[0].norm(dim=-1) * p[0].norm(dim=-1)).sum().clamp_min(1e-9)
    dL = (L - L[0]).abs() / lscale
    return dE, dP, dL


def gaussian_cluster(rng, n, sigma, vfac, mass=None, ref_mass=None):
    pos = np.clip(rng.normal(0.0, sigma, (n, 2)), -29.0, 29.0)
    mass = np.ones(n) if mass is None else mass
    mt = mass.sum()
    vel = rng.normal(0.0, vfac * np.sqrt(mt / (4 * sigma)), (n, 2))
    vel -= (mass[:, None] * vel).sum(0) / mt
    return pos, vel, mass


def rot(theta, x):
    c, s = np.cos(theta), np.sin(theta)
    return x @ np.array([[c, -s], [s, c]]).T


def com_fix(pos, vel, mass):
    mt = mass.sum()
    return pos - (mass[:, None] * pos).sum(0) / mt, vel - (mass[:, None] * vel).sum(0) / mt


def plummer(rng, n, a, mass=None):
    mass = np.ones(n) if mass is None else mass
    u = rng.uniform(0.01, 0.99, n)
    r = np.minimum(a / np.sqrt(u ** (-2 / 3) - 1), 29.0)
    th = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    mt = mass.sum()
    vesc = np.sqrt(2 * mt / np.sqrt(r ** 2 + a ** 2))
    q = np.empty(n)
    for i in range(n):
        while True:
            x, y = rng.uniform(0, 1), rng.uniform(0, 0.1)
            if y < x ** 2 * (1 - x ** 2) ** 3.5:
                q[i] = x
                break
    sp = q * vesc
    ph = rng.uniform(0, 2 * np.pi, n)
    vel = np.stack([sp * np.cos(ph), sp * np.sin(ph)], 1)
    return (*com_fix(pos, vel, mass), mass)


def binary(rng, d, m1, m2, eps=0.5, ecc_scale=1.0):
    """Circular (softened) relative orbit with separation d, COM at rest at origin."""
    mt = m1 + m2
    vrel = np.sqrt(mt * d ** 2 / (d ** 2 + eps ** 2) ** 1.5) * ecc_scale
    pos = np.array([[-d * m2 / mt, 0.0], [d * m1 / mt, 0.0]])
    vel = np.array([[0.0, -vrel * m2 / mt], [0.0, vrel * m1 / mt]])
    th = rng.uniform(0, 2 * np.pi)
    return rot(th, pos), rot(th, vel), np.array([m1, m2])


def scene(spec, seed):
    rng = np.random.default_rng(seed)
    kind, kw = spec["kind"], dict(spec.get("kw", {}))
    out = dict(c=spec.get("c"), eps=spec.get("eps", 0.5), dt=spec.get("dt", 0.1), shift=np.zeros(2))
    n = kw.pop("n", 100)
    if kind == "cluster":
        pos, vel, mass = gaussian_cluster(rng, n, kw.get("sigma", 8.0), kw.get("vfac", 0.3), kw.get("mass"))
    elif kind == "cluster_heavy":
        mh = kw["mh"]
        pos, vel, mass = gaussian_cluster(rng, n, 8.0, kw.get("vfac", 0.3))
        mass = mass.copy()
        mass[0] = mh
        pos[0] = 0.0
        vel[0] = 0.0
        pos, vel = com_fix(pos, vel, mass)
        vt = rng.normal(0.0, kw.get("vfac", 0.3) * np.sqrt(mass.sum() / (4 * 8.0)), (n, 2))
        vel = vt - (mass[:, None] * vt).sum(0) / mass.sum()
    elif kind == "cluster_mass":
        lo, hi = kw["lo"], kw["hi"]
        mass = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        pos, vel, mass = gaussian_cluster(rng, n, 8.0, kw.get("vfac", 0.3), mass)
    elif kind == "cluster_scale":
        mass = np.full(n, kw["m"])
        pos, vel, mass = gaussian_cluster(rng, n, 8.0, kw.get("vfac", 0.3), mass)
    elif kind == "plummer":
        pos, vel, mass = plummer(rng, n, kw["a"])
    elif kind == "binary":
        pos, vel, mass = binary(rng, kw["d"], kw.get("m1", 1.0), kw.get("m2", 1.0), spec.get("eps", 0.5), kw.get("ecc", 1.0))
    elif kind == "pass":
        b, v0, mass_ = kw["b"], kw.get("v0", 1.0), np.array([1.0, 1.0])
        pos = np.array([[-10.0, -b / 2], [10.0, b / 2]])
        vel = np.array([[v0 / 2, 0.0], [-v0 / 2, 0.0]])
        pos, vel, mass = (*[rot(rng.uniform(0, 2 * np.pi), x) for x in (pos, vel)], mass_)
    elif kind == "fall":
        d0 = kw["d0"]
        pos = np.array([[-d0 / 2, 0.0], [d0 / 2, 0.0]])
        vel = np.zeros((2, 2))
        mass = np.ones(2)
        pos, vel = rot(rng.uniform(0, 2 * np.pi), pos), vel
    elif kind == "triple":
        ain, aout = kw["ain"], kw.get("aout", 8.0)
        p1, v1, m1 = binary(rng, ain, 1.0, 1.0, spec.get("eps", 0.5))
        mtot_in = 2.0
        vout = np.sqrt(mtot_in * kw.get("m3", 1.0) / ((mtot_in + kw.get("m3", 1.0)) * aout)) * kw.get("ecc", 1.0)
        pos = np.vstack([p1, [[aout, 0.0]]])
        vel = np.vstack([v1, [[0.0, vout]]])
        mass = np.array([1.0, 1.0, kw.get("m3", 1.0)])
        pos, vel = com_fix(pos, vel, mass)
    elif kind == "shell":
        R = kw["R"]
        th = rng.uniform(0, 2 * np.pi, n)
        r = R + rng.normal(0, 0.5, n)
        pos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        vt = kw.get("vt", 0.0) * np.sqrt(n / R)
        vel = vt * np.stack([-np.sin(th), np.cos(th)], 1) + rng.normal(0, 0.05, (n, 2))
        mass = np.ones(n)
        pos, vel = com_fix(pos, vel, mass)
    elif kind == "disk":
        R = kw["R"]
        r = R * np.sqrt(rng.uniform(0.02, 1.0, n))
        th = rng.uniform(0, 2 * np.pi, n)
        pos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        menc = np.array([(r < x).sum() for x in r]) + 1.0
        vc = np.sqrt(menc / np.sqrt(r ** 2 + 0.25)) * kw.get("vfac", 0.8)
        vel = vc[:, None] * np.stack([-np.sin(th), np.cos(th)], 1)
        mass = np.ones(n)
        pos, vel = com_fix(pos, vel, mass)
    elif kind == "two_cluster":
        h = n // 2
        sep, v0, sg = kw.get("sep", 24.0), kw.get("v0", 1.0), kw.get("sigma", 4.0)
        p1, v1, _ = gaussian_cluster(rng, h, sg, 0.3)
        p2, v2, _ = gaussian_cluster(rng, h, sg, 0.3)
        pos = np.vstack([p1 + [-sep / 2, 0], p2 + [sep / 2, 0]])
        vel = np.vstack([v1 + [v0 / 2, 0], v2 + [-v0 / 2, 0]])
        mass = np.ones(2 * h)
        pos, vel = com_fix(pos, vel, mass)
    elif kind == "uniform":
        L = kw.get("L", 10.0)
        pos = rng.uniform(-L, L, (n, 2))
        vel = rng.normal(0.0, kw.get("v", 0.5), (n, 2))
        mass = np.ones(n)
        pos, vel = com_fix(pos, vel, mass)
    elif kind == "line":
        pos = np.stack([rng.uniform(-15, 15, n), rng.normal(0, 0.2, n)], 1)
        vel = rng.normal(0.0, 0.2, (n, 2))
        mass = np.ones(n)
        pos, vel = com_fix(pos, vel, mass)
    else:
        raise ValueError(kind)
    if "shift" in spec:
        out["shift"] = np.array(spec["shift"], dtype=float)
        pos = pos + out["shift"]
    out.update(pos=pos, vel=vel, mass=mass)
    return out


def specs():
    S = {}

    def add(axis, name, kind, kw=None, **extra):
        S.setdefault(axis, []).append(dict(name=name, kind=kind, kw=kw or {}, **extra))

    add("ref", "cluster_N100_s8_v0.3", "cluster")
    add("ref", "cluster_N300_s8_v0.3", "cluster", dict(n=300))
    add("ref", "bh300_c10", "cluster", dict(n=300), c=10.0)
    for q in (1, 3, 10, 100, 1000, 10000):
        add("mass_ratio_2body", f"binary_q{q}", "binary", dict(d=4.0, m1=1.0, m2=float(q)))
    for mh in (1, 3, 10, 30, 100, 300, 1000, 10000):
        add("mass_dominant", f"cluster_heavy{mh}", "cluster_heavy", dict(mh=float(mh)))
    for lo, hi in ((0.5, 2), (0.1, 10), (0.01, 100), (0.001, 1000)):
        add("mass_spectrum", f"logunif_{lo}_{hi}", "cluster_mass", dict(lo=lo, hi=hi))
    for m in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        add("mass_scale_equal", f"equal_m{m}", "cluster_scale", dict(m=m))
    for v in (0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 8.0):
        add("velocity_scale", f"vfac{v}", "cluster", dict(vfac=v))
    for c in (1.5, 3.0, 5.0, 10.0, 30.0, 100.0, 1000.0):
        add("relativistic_c", f"bh300_c{c}", "cluster", dict(n=300), c=c)
    for v in (0.0, 1.0, 3.0):
        add("relativistic_c", f"bh300_c10_vfac{v}", "cluster", dict(n=300, vfac=v), c=10.0)
    for s in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 16.0, 32.0):
        add("density_sigma", f"sigma{s}", "cluster", dict(sigma=s))
    for a in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        add("density_plummer", f"plummer_a{a}", "plummer", dict(a=a))
    for b in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        add("close_pass_impact", f"pass_b{b}", "pass", dict(b=b))
    for d0 in (1.0, 4.0, 12.0):
        add("close_pass_impact", f"headon_fall_d{d0}", "fall", dict(d0=d0))
    for d in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
        add("binary_tight", f"binary_d{d}", "binary", dict(d=d))
    for a in (0.2, 0.5, 1.0, 2.0):
        add("hier_triple", f"triple_ain{a}", "triple", dict(ain=a))
    add("hier_triple", "triple_ain0.5_m3x10", "triple", dict(ain=0.5, m3=10.0))
    for e in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        add("softening_eps", f"cluster_eps{e}", "cluster", eps=e)
    for e in (0.1, 0.25, 0.5, 1.0):
        add("softening_eps", f"binary_d1_eps{e}", "binary", dict(d=1.0), eps=e)
    for dt in (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4):
        add("dt", f"cluster_dt{dt}", "cluster", dt=dt)
    for dt in (0.05, 0.1, 0.2):
        add("dt", f"bh300_c10_dt{dt}", "cluster", dict(n=300), c=10.0, dt=dt)
    for dt in (0.05, 0.1, 0.2, 0.4):
        add("dt", f"binary_d1_dt{dt}", "binary", dict(d=1.0), dt=dt)
    add("ic_shape", "uniform_L10", "uniform")
    add("ic_shape", "uniform_L25", "uniform", dict(L=25.0))
    add("ic_shape", "shell_R8_rest", "shell", dict(R=8.0))
    add("ic_shape", "shell_R16_rest", "shell", dict(R=16.0))
    add("ic_shape", "shell_R12_rot", "shell", dict(R=12.0, vt=0.3))
    add("ic_shape", "disk_R12", "disk", dict(R=12.0))
    add("ic_shape", "disk_R24", "disk", dict(R=24.0))
    add("ic_shape", "two_cluster_v1", "two_cluster", dict(v0=1.0))
    add("ic_shape", "two_cluster_v4", "two_cluster", dict(v0=4.0))
    add("ic_shape", "two_cluster_v0", "two_cluster", dict(v0=0.0))
    add("ic_shape", "filament", "line")
    for o in (0.0, 8.0, 16.0, 24.0, 32.0, 40.0):
        add("boundary_offset", f"cluster_shift{o}", "cluster", shift=(o, o * 0.5))
    for v in (1.5, 4.0):
        for o in (0.0, 20.0):
            add("boundary_offset", f"cluster_vfac{v}_shift{o}", "cluster", dict(vfac=v), shift=(o, 0.0))
    for m in (0.1, 0.3, 3.0, 10.0, 30.0, 100.0):
        add("mass_dtmatched", f"equal_m{m}_dt0.1/sqrt(m)", "cluster_scale", dict(m=m), tscale=float(m) ** -0.5)
    for mh in (10, 30, 100, 300, 1000):
        add("mass_dtmatched", f"cluster_heavy{mh}_dt*sqrt(10/Mh)", "cluster_heavy", dict(mh=float(mh)), tscale=min(1.0, (10.0 / mh) ** 0.5))
    for q in (10, 100, 1000):
        add("mass_dtmatched", f"binary_q{q}_dt*sqrt(2/(1+q))", "binary", dict(d=4.0, m1=1.0, m2=float(q)), tscale=(2.0 / (1 + q)) ** 0.5)
    for pr in ("fp32", "fp64", "tf32", "half16", "bf16"):
        add("precision", f"cluster_{pr}", "cluster", precision=pr)
        add("precision", f"bh300_c10_{pr}", "cluster", dict(n=300), c=10.0, precision=pr)
    return S


def run_point(model, spec, seed, dev, args, save):
    sc = scene(spec, seed)
    c, eps, dt = sc["c"], sc["eps"], sc["dt"]
    ts = spec.get("tscale", 1.0)
    dt = dt * ts
    sc["dt"] = dt
    steps = int(round(T_TOTAL * ts / dt))
    pos0, vel0, mass = sc["pos"], sc["vel"], sc["mass"]
    n = len(mass)
    prec = spec.get("precision", "fp32")
    dtype = torch.float64 if prec == "fp64" else torch.float32
    Pt, Vt = truth(pos0, vel0, mass, steps, dt, eps, c, dev)
    rng = np.random.default_rng(seed + 1)
    L = max(float(np.sqrt((pos0 ** 2).sum(1).mean())), 1.0)
    Pp, _ = truth(pos0 + 1e-5 * L * rng.normal(size=pos0.shape), vel0, mass, steps, dt, eps, c, dev)
    m_t = torch.tensor(mass, dtype=torch.float64, device=dev)
    ticks = {t: int(round(t * ts / dt)) for t in TIMES}
    if prec != "fp32":
        model._graphs = {}
        model._kf = None
    if prec == "fp64":
        model.double()
    if prec == "half16":
        sf.apply_opt(model, "half16")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    if prec == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    amp = prec == "bf16"
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            Pm, Vm, nonfinite = model_rollout(model, pos0, vel0, mass, steps, dt, c, dev, dtype)
    finally:
        if prec == "fp64":
            model.float()
        if prec == "half16":
            model.opt.discard("half16")
        if prec != "fp32":
            model._graphs = {}
            model._kf = None
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_tf32
    T = Pm.shape[0]
    Ptr = Pt[:T]
    err_t = (Pm - Ptr).norm(dim=-1).mean(1).cpu().numpy()
    t_axis = np.arange(T) * dt
    ball = (torch.tensor(pos0, dtype=torch.float64, device=dev)[None] + torch.tensor(t_axis, device=dev)[:, None, None] * torch.tensor(vel0, dtype=torch.float64, device=dev)[None])
    ball_err = (ball - Ptr).norm(dim=-1).mean(1).cpu().numpy()
    floor_err = (Pp[:T] - Ptr).norm(dim=-1).mean(1).cpu().numpy()
    iheavy = int(np.argmax(mass))
    heavy_err = (Pm[:, iheavy] - Ptr[:, iheavy]).norm(dim=-1).cpu().numpy()
    dEm, dPm, dLm = conserved(Pm, Vm, m_t, eps, c)
    dEt, dPt, dLt = conserved(Pt, Vt, m_t, eps, c)
    fin = np.isfinite(err_t)
    out = {"n": n, "seed": seed, "steps": steps, "dt": dt, "nonfinite_tick": nonfinite}
    for t, k in ticks.items():
        out[f"err@{t}"] = float(err_t[k]) if k < T else float("nan")
        out[f"floor@{t}"] = float(floor_err[k]) if k < T else float("nan")
        out[f"ballistic@{t}"] = float(ball_err[k]) if k < T else float("nan")
        out[f"heavy_err@{t}"] = float(heavy_err[k]) if k < T else float("nan")
    out["err@10"] = float(err_t[-1]) if T == steps + 1 else float("nan")
    out["floor@10"] = float(floor_err[-1])
    bad = np.where(~fin | (err_t > 1.0))[0]
    out["t_err_gt_1"] = float(bad[0] * dt) if len(bad) else None
    worse = np.where(err_t > ball_err * 1.0)[0]
    worse = worse[worse > 0]
    out["t_worse_than_ballistic"] = float(worse[0] * dt) if len(worse) else None
    out["dE_model"], out["dE_true"] = float(dEm[-1]), float(dEt[-1])
    out["dE_model_max"], out["dE_true_max"] = float(dEm.max()), float(dEt.max())
    out["dP_model"], out["dP_true"] = float(dPm.max()), float(dPt.max())
    out["dL_model"], out["dL_true"] = float(dLm.max()), float(dLt.max())
    out["vmax_over_c_model"] = float(Vm.norm(dim=-1).max() / c) if c else None
    out["frac_outside_grid_max"] = float(((Pm.abs() > EXTENT / 2).any(-1).double().mean(1)).max())
    out["frac_unbound_t0"] = float(((0.5 * (Vt[0] ** 2).sum(-1) - (m_t[None, :] / ((Pt[0][:, None] - Pt[0][None]).norm(dim=-1) + eps)).sum(1)) > 0).double().mean()) if c is None else None
    out["min_pair_sep_truth"] = float(min_sep(Pt, n))
    if save is not None and seed == 4738:
        save[spec["axis"] + "/" + spec["name"] + "/model"] = Pm.float().cpu().numpy()
        save[spec["axis"] + "/" + spec["name"] + "/truth"] = Ptr.float().cpu().numpy()
        save[spec["axis"] + "/" + spec["name"] + "/err"] = err_t
        save[spec["axis"] + "/" + spec["name"] + "/ballistic"] = ball_err
        save[spec["axis"] + "/" + spec["name"] + "/floor"] = floor_err
        save[spec["axis"] + "/" + spec["name"] + "/dt"] = np.array(dt)
    return out


def min_sep(P, n):
    if n < 2:
        return float("nan")
    best = float("inf")
    for t in range(0, P.shape[0], max(1, P.shape[0] // 25)):
        d = torch.cdist(P[t], P[t])
        d.fill_diagonal_(float("inf"))
        best = min(best, float(d.min()))
    return best


def build_model(args, dev):
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    for o in [x for x in args.opt.split(",") if x]:
        sf.apply_opt(model, o)
    return model


def summarize(per_seed):
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], float) or per_seed[0][k] is None]
    mean = {}
    for k in keys:
        vals = [r[k] for r in per_seed if r[k] is not None]
        mean[k] = float(np.nanmean(vals)) if vals and not all(np.isnan(vals)) else None
    mean["n_nonfinite_seeds"] = int(sum(r["nonfinite_tick"] is not None for r in per_seed))
    nf = [r["nonfinite_tick"] for r in per_seed if r["nonfinite_tick"] is not None]
    mean["first_nonfinite_tick"] = min(nf) if nf else None
    tb = [r["t_err_gt_1"] for r in per_seed]
    mean["t_err_gt_1_median"] = float(np.median([t if t is not None else 99.0 for t in tb]))
    return mean


def selftest(dev):
    rng = np.random.default_rng(4738)
    pos, vel, mass = gaussian_cluster(rng, 100, 8.0, 0.3)
    for c in (None, 10.0):
        P, V = truth(pos, vel, mass, 10, 0.1, 0.5, c, dev)
        Pg, Vg = gs.rollout_torch(pos, vel, 10, dev, dt=0.1, substeps=4, eps=0.5, relativistic=c is not None, c=c or 1.0)
        print(f"selftest truth vs gs.rollout_torch c={c}: max diff {float(np.abs(P.cpu().numpy() - Pg).max()):.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axes", default="")
    ap.add_argument("--opt", default="kcache,fastio,cl,graph")
    ap.add_argument("--traj", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    selftest(dev)
    model = build_model(args, dev)
    S = specs()
    axes = [a for a in args.axes.split(",") if a] or list(S)
    save = {} if args.traj else None
    res = {"meta": {"ckpt": args.ckpt, "opt": args.opt, "seeds": list(SEEDS), "times": list(TIMES), "eps_default": 0.5, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(dev), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, "axes": {}}
    for ax in axes:
        res["axes"][ax] = {}
        for spec in S[ax]:
            spec["axis"] = ax
            per_seed = []
            for seed in SEEDS:
                try:
                    per_seed.append(run_point(model, spec, seed, dev, args, save))
                except Exception as e:
                    print(f"FAIL {ax}/{spec['name']} seed {seed}: {type(e).__name__} {str(e)[:200]}", flush=True)
                    per_seed.append(None)
                    torch.cuda.empty_cache()
            ok = [r for r in per_seed if r is not None]
            if not ok:
                res["axes"][ax][spec["name"]] = {"error": "all seeds failed"}
                continue
            m = summarize(ok)
            res["axes"][ax][spec["name"]] = {"mean": m, "per_seed": ok, "n_failed_seeds": len(per_seed) - len(ok)}
            print(f"{ax:18s} {spec['name']:26s} err@.5/1/2/5 {[round(m[f'err@{t}'], 4) for t in TIMES]} floor@5 {m['floor@5.0']:.1e} ballistic@5 {m['ballistic@5.0']:.3g} "
                  f"dE {m['dE_model_max']:.3g} (true {m['dE_true_max']:.3g}) nonfinite {m['first_nonfinite_tick']} t>1 {m['t_err_gt_1_median']}", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
    json.dump(res, open(args.out, "w"), indent=1)
    if save:
        np.savez_compressed(args.traj, **save)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
