"""Star-plus-planet and star-plus-disk scenes at a controlled star:body mass ratio q, plus orbit-element metrics.
Newtonian, G=1, softening 0.5, dt 0.1, 2D, truth from scatter_field.rollout_mass_torch (float64, 4 substeps per tick).
Every orbit has period >= PERIOD_FLOOR ticks (a body must not cross more than a few grid cells per tick). Scenes are lists of
(P, V, m) in absolute coordinates (CENTER offset) like scatter_ft_data. Design and pass/kill lines: docs/theory/mass-conditioned-scatter-field.md."""
import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_ft_data as fd

C = sf.CENTER
STAGES = (1, 3, 10, 30, 100, 300, 1000)
PERIOD_FLOOR = 40
DT = 0.1
EPS = 0.5
A_MAX = 24.0
R_DISK_MAX = 20.0
A_PLANET_IN_DISK_MAX = 12.0
M_STAR = (20.0, 1000.0)


def log_uniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def a_min(mu, floor=PERIOD_FLOOR, dt=DT):
    return (mu * (floor * dt / (2 * np.pi)) ** 2) ** (1.0 / 3.0)


def circ_speed(M, r, eps=EPS):
    return np.sqrt(M * r ** 2 / (r ** 2 + eps ** 2) ** 1.5)


def com_frame(pos, vel, mass):
    pos = pos - (mass[:, None] * pos).sum(0) / mass.sum()
    return pos, vel - (mass[:, None] * vel).sum(0) / mass.sum()


def planet_orbit(rng, M, m, a=None, e=None):
    """Relative state of a body of mass m about a star of mass M, started at apocentre. Returns (rel_pos, rel_vel, a, e)."""
    mu = M + m
    lo = max(a_min(mu), 3.0)
    e = rng.uniform(0.0, 0.3) if e is None else e
    a = log_uniform(rng, lo, A_MAX / (1 + e)) if a is None else a
    ra = a * (1 + e)
    va = np.sqrt(mu * (1 - e) / ra) * np.sqrt(ra ** 2 / (ra ** 2 + EPS ** 2)) ** 1.5
    th = rng.uniform(0, 2 * np.pi)
    sgn = rng.choice([-1.0, 1.0])
    rhat, that = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    return ra * rhat, sgn * va * that, a, e


def star_planet(rng, q, M=None):
    M = log_uniform(rng, *M_STAR) if M is None else M
    m = M / q
    rp, vp, a, e = planet_orbit(rng, M, m)
    pos = np.stack([np.zeros(2), rp])
    vel = np.stack([np.zeros(2), vp])
    mass = np.array([M, m])
    pos, vel = com_frame(pos, vel, mass)
    return pos + rng.uniform(-2, 2, 2), vel, mass


def star_disk(rng, q, n_disk=None, planet=True, M=None, r_ratio=(1.0, 10.0), disk_mass_cap=0.1):
    """Star (index 0), optional planet of mass M/q (index 1), n_disk light bodies of mass M/(q*R), R log-uniform in r_ratio, total disk mass <= disk_mass_cap*M."""
    M = log_uniform(rng, *M_STAR) if M is None else M
    n_disk = int(rng.integers(30, 201)) if n_disk is None else n_disk
    m_pl = M / q
    m_d = min(M / (q * log_uniform(rng, *r_ratio)), disk_mass_cap * M / n_disk)
    r_in = max(a_min(M), 6.0)
    r = rng.uniform(r_in, R_DISK_MAX, n_disk)
    th = rng.uniform(0, 2 * np.pi, n_disk)
    vc = circ_speed(M, r) * (1 + 0.02 * rng.normal(size=n_disk))
    dpos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    dvel = vc[:, None] * np.stack([-np.sin(th), np.cos(th)], 1) + 0.02 * vc[:, None] * rng.normal(size=(n_disk, 2))
    pos, vel, mass = [np.zeros((1, 2))], [np.zeros((1, 2))], [M]
    if planet:
        e_p = rng.uniform(0.0, 0.1)
        rp, vp, _, _ = planet_orbit(rng, M, m_pl, a=log_uniform(rng, max(a_min(M + m_pl), 3.0), A_PLANET_IN_DISK_MAX), e=e_p)
        rp, vp = np.abs(np.linalg.norm(rp)) * np.array([1.0, 0.0]), np.linalg.norm(vp) * np.array([0.0, 1.0])
        pos.append(rp[None])
        vel.append(vp[None])
        mass.append(m_pl)
    pos = np.concatenate(pos + [dpos])
    vel = np.concatenate(vel + [dvel])
    mass = np.concatenate([np.array(mass), np.full(n_disk, m_d)])
    pos, vel = com_frame(pos, vel, mass)
    return pos, vel, mass


def scenes(num, steps, seed, device, q, kind="mixed", dt=DT, eps=EPS):
    """q: float, or (lo, hi) for a log-uniform ratio per scene. kind: planet | disk | mixed (half and half)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(num):
        qi = float(log_uniform(rng, *q)) if isinstance(q, (tuple, list)) else float(q)
        k = kind if kind != "mixed" else ("planet", "disk")[i % 2]
        pos, vel, mass = star_planet(rng, qi) if k == "planet" else star_disk(rng, qi)
        out.append(fd.truth(pos, vel, mass, steps, device, dt=dt, eps=eps))
    return out


def stage_groups(stage, num, steps, seed, device, replay_base=0.3):
    """Scene groups for curriculum stage index (0..len(STAGES)-1): (name, data, weight). The current ratio gets 0.45, earlier ratios
    share 0.25, unit-mass base scenes (E-style) keep replay_base of the draws so the in-distribution regime is not traded away."""
    out = [(f"q{STAGES[stage]:g}", scenes(num, steps, seed + stage, device, STAGES[stage]), 0.45 if stage else 0.7)]
    if stage:
        out.append((f"q1-{STAGES[stage - 1]:g}", scenes(num, steps, seed + 100 + stage, device, (1.0, float(STAGES[stage - 1]))), 0.25))
    out.append(("unit_base", fd.base(num, steps, seed + 200, device), replay_base))
    return out


def orbit_elements(pos, vel, mass, star=0, g=1.0):
    """Semi-major axis and eccentricity of every non-star body about the star (two-body osculating, mu = g (M + m_i)); numpy (N,2)."""
    r = pos - pos[star]
    v = vel - vel[star]
    idx = np.array([i for i in range(len(mass)) if i != star])
    r, v, m = r[idx], v[idx], mass[idx]
    mu = g * (mass[star] + m)
    rn = np.linalg.norm(r, axis=-1)
    a = 1.0 / (2.0 / rn - (v ** 2).sum(-1) / mu)
    h = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    ev = np.stack([v[:, 1] * h, -v[:, 0] * h], -1) / mu[:, None] - r / rn[:, None]
    return a, np.linalg.norm(ev, axis=-1)


@torch.no_grad()
def orbit_metrics(model, data, device, ticks=(20, 50, 100), eps=EPS):
    """Roll the model from each scene's start and compare with the stored truth. Star = index 0. Per scene: relative |a - a_truth|/a_truth
    and |e - e_truth| of the planet (index 1, planet scenes) or the median over disk bodies, mean light-body position error, relative energy
    drift of the model and of the truth (softened energy, same definition). Returns medians over scenes per tick plus bound/finite flags."""
    P, V, M, mask = [t.float().to(device) for t in sf.to_tensors(data, device)]
    T = max(ticks)
    Pm, Vm = sf.rollout(model, P[:, 0], V[:, 0], M, mask, T)
    res = {}
    for k in ticks:
        da, de, perr, dEm, dEt, fin = [], [], [], [], [], []
        for s in range(P.shape[0]):
            n = int(mask[s].sum())
            m = M[s, :n].double().cpu().numpy()
            pt, vt = P[s, k, :n].double().cpu().numpy(), V[s, k, :n].double().cpu().numpy()
            pm, vm = Pm[s, k, :n].double().cpu().numpy(), Vm[s, k, :n].double().cpu().numpy()
            ok = bool(np.isfinite(pm).all() and np.isfinite(vm).all())
            fin.append(ok)
            perr.append(float(np.linalg.norm(pm[1:] - pt[1:], axis=-1).mean()) if ok else np.inf)
            if ok:
                at, et = orbit_elements(pt, vt, m)
                am, em = orbit_elements(pm, vm, m)
                bound = at > 0
                da.append(float(np.median((np.abs(am - at) / np.abs(at))[bound])) if bound.any() else np.nan)
                de.append(float(np.median(np.abs(em - et)[bound])) if bound.any() else np.nan)
            else:
                da.append(np.inf)
                de.append(np.inf)
            for tag, PP, VV, store in (("m", Pm, Vm, dEm), ("t", P, V, dEt)):
                e0 = float(sf.energy(PP[s:s + 1, 0], VV[s:s + 1, 0], M[s:s + 1], mask[s:s + 1], eps))
                e1 = float(sf.energy(PP[s:s + 1, k], VV[s:s + 1, k], M[s:s + 1], mask[s:s + 1], eps))
                store.append(abs(e1 - e0) / abs(e0))
        res[k] = dict(da_over_a=float(np.nanmedian(da)), de=float(np.nanmedian(de)), pos_err=float(np.median(perr)),
                      dE_model=float(np.nanmedian(dEm)), dE_truth=float(np.nanmedian(dEt)), dE_model_max=float(np.nanmax(dEm)),
                      finite=bool(all(fin)))
    return res
