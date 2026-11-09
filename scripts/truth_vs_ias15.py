import argparse
import json
import time

import numpy as np
import rebound
import torch

from scatter_field import rollout_mass_torch

EPS = 0.5
DT = 0.1


def torch_acc(p, m, eps):
    d = p[None, :, :] - p[:, None, :]
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
    np.fill_diagonal(inv, 0.0)
    return (d * inv[..., None] * m[None, :, None]).sum(1)


def make_sim(pos, vel, m, eps):
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.softening = eps
    for i in range(len(m)):
        sim.add(m=float(m[i]), x=float(pos[i, 0]), y=float(pos[i, 1]), z=0.0, vx=float(vel[i, 0]), vy=float(vel[i, 1]), vz=0.0)
    sim.move_to_com()
    return sim


def ias15_rollout(pos, vel, m, steps, eps, dt=DT):
    sim = make_sim(pos, vel, m, eps)
    ps, vs = [], []
    for k in range(steps + 1):
        sim.integrate(k * dt, exact_finish_time=1)
        ps.append(np.array([[p.x, p.y] for p in sim.particles]))
        vs.append(np.array([[p.vx, p.vy] for p in sim.particles]))
    return np.stack(ps), np.stack(vs)


def truth_rollout(pos, vel, m, steps, eps, substeps):
    pos = pos - (m[:, None] * pos).sum(0) / m.sum()
    vel = vel - (m[:, None] * vel).sum(0) / m.sum()
    return rollout_mass_torch(pos, vel, m, steps, "cpu", dt=DT, substeps=substeps, eps=eps)


def energy(P, V, m, eps):
    K = 0.5 * (m[None, :] * (V ** 2).sum(-1)).sum(1)
    d = P[:, None, :, :] - P[:, :, None, :]
    r = np.sqrt((d ** 2).sum(-1) + eps ** 2)
    iu = np.triu_indices(len(m), 1)
    U = -(m[None, iu[0]] * m[None, iu[1]] / r[:, iu[0], iu[1]]).sum(1)
    return K + U, K[0] + abs(U[0])


def lagrangian(P, m, qs=(0.1, 0.5, 0.9)):
    out = []
    for p in P:
        c = (m[:, None] * p).sum(0) / m.sum()
        r = np.sqrt(((p - c) ** 2).sum(1))
        o = np.argsort(r)
        cm = np.cumsum(m[o]) / m.sum()
        out.append([r[o][np.searchsorted(cm, q)] for q in qs])
    return np.array(out)


def force_check():
    rows = []
    for eps in (0.5, 0.1):
        for d in (0.05, 0.2, 0.5, 1.0, 3.0):
            pos = np.array([[0.0, 0.0], [d, 0.0]])
            m = np.array([1.0, 0.3])
            sim = rebound.Simulation()
            sim.G = 1.0
            sim.integrator = "ias15"
            sim.softening = eps
            sim.add(m=1.0, x=0, y=0, z=0)
            sim.add(m=0.3, x=d, y=0, z=0)
            sim.integrate(1e-12)
            pr = sim.particles
            a_reb = np.array([[pr[0].ax, pr[0].ay], [pr[1].ax, pr[1].ay]])
            p_now = np.array([[pr[0].x, pr[0].y], [pr[1].x, pr[1].y]])
            a_aug = torch_acc(p_now, m, eps)
            plummer = 0.3 * d / (d * d + eps * eps) ** 1.5
            rows.append(dict(eps=eps, d=d, a0_rebound=float(a_reb[0, 0]), a0_augur=float(a_aug[0, 0]), a0_plummer=float(plummer), rel_err=float(abs(a_reb[0, 0] - a_aug[0, 0]) / abs(a_aug[0, 0])), rel_err_plummer=float(abs(a_reb[0, 0] - plummer) / plummer)))
    return rows


def kepler_ics(masses, a_list, e_list):
    n = len(masses)
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    for i in range(1, n):
        a, e = a_list[i - 1], e_list[i - 1]
        mu = masses[0] + masses[i]
        r = a * (1 + e)
        v = np.sqrt(mu * (1 - e) / (a * (1 + e)))
        ang = 0.0 if i == 1 else 1.3
        pos[i] = r * np.array([np.cos(ang), np.sin(ang)])
        vel[i] = v * np.array([-np.sin(ang), np.cos(ang)])
    return pos, vel


def compare(name, pos, vel, m, steps, eps):
    t0 = time.time()
    Pi, Vi = ias15_rollout(pos, vel, m, steps, eps)
    ti = time.time() - t0
    Pt4, Vt4 = truth_rollout(pos, vel, m, steps, eps, 4)
    Pt16, Vt16 = truth_rollout(pos, vel, m, steps, eps, 16)
    Ei, sc = energy(Pi, Vi, m, eps)
    energy_t4, _ = energy(Pt4, Vt4, m, eps)
    energy_t16, _ = energy(Pt16, Vt16, m, eps)
    res = dict(name=name, N=len(m), steps=steps, eps=eps, ias15_s=ti,
               dE_ias15_max=float(np.abs(Ei - Ei[0]).max() / sc),
               dE_t4_max=float(np.abs(energy_t4 - energy_t4[0]).max() / sc),
               dE_t16_max=float(np.abs(energy_t16 - energy_t16[0]).max() / sc),
               dE_t4_vs_ias15_max=float(np.abs(energy_t4 - Ei).max() / sc),
               dE_t16_vs_ias15_max=float(np.abs(energy_t16 - Ei).max() / sc),
               dE_t4_vs_t16_max=float(np.abs(energy_t4 - energy_t16).max() / sc),
               pos_t4_vs_ias15_max=float(np.abs(Pt4 - Pi).max()),
               pos_t16_vs_ias15_max=float(np.abs(Pt16 - Pi).max()),
               pos_t4_vs_t16_max=float(np.abs(Pt4 - Pt16).max()))
    return res, (Pi, Pt4, Pt16)


def plummer_ic(n, seed, mtot=30.0, a=8.0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n)
    r = a * np.sqrt(u / (1 - u + 1e-12))
    r = np.minimum(r, 10 * a)
    th = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    m = np.full(n, mtot / n)
    vel = rng.normal(size=(n, 2))
    pos -= (m[:, None] * pos).sum(0) / m.sum()
    vel -= (m[:, None] * vel).sum(0) / m.sum()
    d = pos[None] - pos[:, None]
    rr = np.sqrt((d ** 2).sum(-1) + EPS ** 2)
    iu = np.triu_indices(n, 1)
    U = -(m[iu[0]] * m[iu[1]] / rr[iu]).sum()
    K = 0.5 * (m * (vel ** 2).sum(1)).sum()
    vel *= np.sqrt(0.5 * abs(U) / K)
    return pos, vel, m


def plummer_test(n, seed, steps, nrep, tol_scale):
    pos, vel, m = plummer_ic(n, seed)
    res, (Pi, Pt4, Pt16) = compare(f"plummer-n{n}-seed{seed}", pos, vel, m, steps, EPS)
    L_i, L4, L16 = lagrangian(Pi, m), lagrangian(Pt4, m), lagrangian(Pt16, m)
    reps = []
    rng = np.random.default_rng(seed + 1000)
    for k in range(nrep):
        Pr, _ = ias15_rollout(pos + 1e-8 * rng.normal(size=pos.shape), vel, m, steps, EPS)
        reps.append(lagrangian(Pr, m))
    reps = np.stack(reps)
    spread = reps.std(0, ddof=1)
    checks = [steps // 4, steps // 2, steps]
    lag = {}
    for q, qn in enumerate(("r10", "r50", "r90")):
        for c in checks:
            lag[f"{qn}@{c}"] = dict(ias15=float(L_i[c, q]), t4=float(L4[c, q]), t16=float(L16[c, q]),
                                    replica_std=float(spread[c, q]), abs_t4_vs_ias15=float(abs(L4[c, q] - L_i[c, q])),
                                    abs_t16_vs_ias15=float(abs(L16[c, q] - L_i[c, q])),
                                    within_2std=bool(abs(L4[c, q] - L_i[c, q]) <= 2 * spread[c, q]))
    res["lagrangian"] = lag
    res["r_replica_std_max_over_time"] = spread.max(0).tolist()
    return res


def substep_convergence(n, seeds, steps):
    out = []
    for seed in seeds:
        pos, vel, m = plummer_ic(n, seed)
        Pi, Vi = ias15_rollout(pos, vel, m, steps, EPS)
        Ei, sc = energy(Pi, Vi, m, EPS)
        Li = lagrangian(Pi, m)
        for sub in (1, 2, 4, 8, 16, 32):
            P, V = truth_rollout(pos, vel, m, steps, EPS, sub)
            E, _ = energy(P, V, m, EPS)
            L = lagrangian(P, m)
            out.append(dict(n=n, seed=seed, substeps=sub, pos_vs_ias15_max=float(np.abs(P - Pi).max()),
                            dE_vs_ias15_max=float(np.abs(E - Ei).max() / sc),
                            lagrangian_abs_diff_final=[float(x) for x in np.abs(L[-1] - Li[-1])],
                            lagrangian_rel_diff_final=[float(x) for x in np.abs(L[-1] - Li[-1]) / Li[-1]]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["force", "kepler", "plummer", "substep-convergence"], required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "force":
        out = force_check()
    elif a.stage == "kepler":
        out = []
        for eps in (EPS, 1e-3):
            pos, vel = kepler_ics(np.array([10.0, 0.1]), [5.0], [0.5])
            out.append(compare(f"kepler-e0.5-n2-eps{eps}", pos, vel, np.array([10.0, 0.1]), 500, eps)[0])
            m3 = np.array([10.0, 0.3, 0.1])
            pos, vel = kepler_ics(m3, [5.0, 12.0], [0.5, 0.5])
            out.append(compare(f"kepler-e0.5-n3-eps{eps}", pos, vel, m3, 500, eps)[0])
    elif a.stage == "substep-convergence":
        out = substep_convergence(a.n, [a.seed + k for k in range(4)], a.steps)
    else:
        out = plummer_test(a.n, a.seed, a.steps, a.replicas, None)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
