"""Training-scene generators for the scatter-field fine-tune fixes (decisions #4/#6, scatter-review-decisions-2026-10-06.md).

Every generator returns a list of (P, V, m) trajectories (T, N, 2) in absolute coordinates (CENTER offset) with exact softened float64 truth
(4 substeps per tick, as in training). Newtonian, G=1, dt 0.1, open boundary.
"""
import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts.scatter_ood import plummer

C = sf.CENTER


def truth(pos, vel, mass, steps, device, dt=0.1, eps=0.5):
    P, V = sf.rollout_mass_torch(pos + C, vel, mass, steps, device, dt=dt, eps=eps)
    return P, V, mass


def com_fix(vel, mass):
    return vel - (mass[:, None] * vel).sum(0) / mass.sum()


def masses(rng, n, kind, lo, hi):
    if kind == "unit":
        return np.ones(n)
    if kind == "spec":
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    if kind == "scale":
        return np.full(n, float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
    m = np.ones(n)
    m[0] = float(np.exp(rng.uniform(np.log(3.0), np.log(hi))))
    return m


def box_scene(rng, n, mass):
    pos, vel = gs.init_bodies(n, rng, scale=True)
    vel = vel * np.sqrt(mass.mean())
    return pos - C, com_fix(vel, mass)


def cluster_scene(rng, n, mass, sigma, vfac):
    pos = np.clip(rng.normal(0.0, sigma, (n, 2)), -28.0, 28.0)
    vel = rng.normal(0.0, vfac * np.sqrt(mass.sum() / (4 * sigma)), (n, 2))
    return pos, com_fix(vel, mass)


def mixed_scene(rng, kind, lo=1.0, hi=1.0, n_box=(10, 200), n_cl=(80, 300), sigma=(5.0, 10.0), vfac=(0.1, 0.6), p_cluster=0.5):
    if rng.uniform() < p_cluster:
        n = int(rng.integers(n_cl[0], n_cl[1] + 1))
        m = masses(rng, n, kind, lo, hi)
        sg = float(np.exp(rng.uniform(np.log(sigma[0]), np.log(sigma[1]))))
        return (*cluster_scene(rng, n, m, sg, rng.uniform(*vfac)), m)
    n = int(rng.integers(n_box[0], n_box[1] + 1))
    m = masses(rng, n, kind, lo, hi)
    return (*box_scene(rng, n, m), m)


def base(num, steps, seed, device, eps=0.5):
    rng = np.random.default_rng(seed)
    return [truth(*mixed_scene(rng, "unit"), steps, device, eps=eps) for _ in range(num)]


def mass_spectrum(num, steps, seed, device, lo, hi, frac_unit=0.3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        if rng.uniform() < frac_unit:
            kind = "unit"
        else:
            kind = ("spec", "scale", "dom")[int(rng.integers(3))]
        out.append(truth(*mixed_scene(rng, kind, lo, hi), steps, device))
    return out


def eps_scenes(num, steps, seed, device, eps):
    rng = np.random.default_rng(seed)
    return [truth(*mixed_scene(rng, "unit"), steps, device, eps=eps) for _ in range(num)]


def dense(num, steps, seed, device, frac_base=0.4):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        u = rng.uniform()
        if u < frac_base:
            out.append(truth(*mixed_scene(rng, "unit"), steps, device))
        elif rng.uniform() < 0.5:
            n = int(rng.integers(80, 301))
            m = np.ones(n)
            sg = float(np.exp(rng.uniform(np.log(1.0), np.log(5.0))))
            out.append(truth(*cluster_scene(rng, n, m, sg, rng.uniform(0.1, 0.6)), m, steps, device))
        else:
            n = int(rng.integers(100, 301))
            a = float(np.exp(rng.uniform(np.log(0.25), np.log(2.0))))
            pos, vel, m = plummer(rng, n, a)
            out.append(truth(pos, vel, m, steps, device))
    return out


def large_n(num, steps, seed, device, n_range):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        n = int(np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1]))))
        m = np.ones(n)
        sg = rng.uniform(5.0, 10.0)
        out.append(truth(*cluster_scene(rng, n, m, sg, rng.uniform(0.1, 1.0)), m, steps, device))
    return out
