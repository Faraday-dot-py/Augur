"""Star plus planetesimal disk vs direct N-body. Modes:
  real  : reference only (units AU, Msun, yr; G=4 pi^2). Learned models are unit-mass, so out of range.
  proxy : equal-mass in-range proxy (Plummer-core "star" + circular disk), reference vs CentralForceDynamics.
Pre-registration: docs/debugging/experiment-log.md (star plus planetesimal disk entry)."""
import argparse
import json
import sys
import time

import numpy as np
import torch

G_REAL = 4 * np.pi ** 2


def accel(pos, mass, eps, g):
    d = pos[None, :, :] - pos[:, None, :]
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5
    inv.fill_diagonal_(0.0)
    return g * (d * (inv * mass[None, :])[..., None]).sum(1)


def energy(pos, vel, mass, eps, g):
    d = pos[None, :, :] - pos[:, None, :]
    r = ((d ** 2).sum(-1) + eps ** 2).sqrt()
    iu = torch.triu_indices(len(pos), len(pos), 1, device=pos.device)
    mm = mass[iu[0]] * mass[iu[1]]
    return float(0.5 * (mass[:, None] * vel ** 2).sum() - g * (mm / r[iu[0], iu[1]]).sum())


def ang_mom(pos, vel, mass):
    c = torch.cross(pos, vel, dim=-1) if pos.shape[1] == 3 else (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
    return (mass[:, None] * c).sum(0) if pos.shape[1] == 3 else (mass * c).sum()


def leapfrog(pos, vel, mass, eps, g, h, nsteps, every, first_dense=0, log=None):
    snaps_p, snaps_v, ticks = [], [], []
    a = accel(pos, mass, eps, g)
    e0, l0 = energy(pos, vel, mass, eps, g), ang_mom(pos, vel, mass)
    diag = []
    for s in range(1, nsteps + 1):
        vel = vel + 0.5 * h * a
        pos = pos + h * vel
        a = accel(pos, mass, eps, g)
        vel = vel + 0.5 * h * a
        if s % every == 0 or s <= first_dense:
            snaps_p.append(pos.cpu().numpy().astype(np.float32))
            snaps_v.append(vel.cpu().numpy().astype(np.float32))
            ticks.append(s)
        if s % max(nsteps // 20, 1) == 0:
            de = (energy(pos, vel, mass, eps, g) - e0) / abs(e0)
            dl = float(((ang_mom(pos, vel, mass) - l0).norm() / l0.norm()))
            diag.append((s, de, dl))
            if log:
                print(f"step {s} dE/E0 {de:.3e} dL/L0 {dl:.3e}", flush=True)
    return np.stack(snaps_p), np.stack(snaps_v), np.array(ticks), diag, (e0, l0)


def kepler_E(M, e):
    E = M + e * np.sin(M)
    for _ in range(30):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def real_ic(rng, n_disk, jup, r_in=0.3, r_out=30.0, disk_frac=0.015, m_jup=1e-3):
    u = rng.uniform(size=n_disk)
    a = (r_in ** 0.5 + u * (r_out ** 0.5 - r_in ** 0.5)) ** 2
    e = rng.rayleigh(0.008, n_disk)
    inc = rng.rayleigh(0.004, n_disk)
    om = rng.uniform(0, 2 * np.pi, n_disk)
    Om = rng.uniform(0, 2 * np.pi, n_disk)
    M = rng.uniform(0, 2 * np.pi, n_disk)
    ja = np.array(jup)
    a = np.concatenate([ja, a])
    e = np.concatenate([np.full(len(ja), 0.0), e])
    inc = np.concatenate([np.zeros(len(ja)), inc])
    om = np.concatenate([np.zeros(len(ja)), om])
    Om = np.concatenate([np.zeros(len(ja)), Om])
    M = np.concatenate([rng.uniform(0, 2 * np.pi, len(ja)), M])
    mu = G_REAL
    E = kepler_E(M, e)
    xo, yo = a * (np.cos(E) - e), a * np.sqrt(1 - e ** 2) * np.sin(E)
    n = np.sqrt(mu / a ** 3)
    vxo, vyo = -a * n * np.sin(E) / (1 - e * np.cos(E)), a * n * np.sqrt(1 - e ** 2) * np.cos(E) / (1 - e * np.cos(E))

    def rot(x, y):
        x1, y1 = x * np.cos(om) - y * np.sin(om), x * np.sin(om) + y * np.cos(om)
        y2, z2 = y1 * np.cos(inc), y1 * np.sin(inc)
        return np.stack([x1 * np.cos(Om) - y2 * np.sin(Om), x1 * np.sin(Om) + y2 * np.cos(Om), z2], -1)

    pos, vel = rot(xo, yo), rot(vxo, vyo)
    m_d = np.full(n_disk, disk_frac / n_disk)
    mass = np.concatenate([[1.0], np.full(len(ja), m_jup), m_d])
    pos = np.concatenate([np.zeros((1, 3)), pos])
    vel = np.concatenate([np.zeros((1, 3)), vel])
    pos -= (mass[:, None] * pos).sum(0) / mass.sum()
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    return pos, vel, mass


def helio_elements(pos, vel, mass, g):
    r = pos[:, 1:] - pos[:, :1]
    v = vel[:, 1:] - vel[:, :1]
    mu = g * (mass[0] + mass[1:])
    rn = np.linalg.norm(r, axis=-1)
    a = 1.0 / (2.0 / rn - (v ** 2).sum(-1) / mu)
    h = np.cross(r, v)
    ev = np.cross(v, h) / mu[..., None] - r / rn[..., None]
    return a, np.linalg.norm(ev, axis=-1)


def merge_count(snap_pos, mass, movable, r_merge, star=None, r_star=None):
    """Greedy sticky accretion on saved snapshots. movable: indices eligible. Returns alive mask, merged mass, counts."""
    mass = mass.copy()
    alive = np.zeros(len(mass), bool)
    alive[movable] = True
    n_merge = 0
    n_star = 0
    for p in snap_pos:
        idx = np.nonzero(alive)[0]
        if star is not None:
            d = np.linalg.norm(p[idx] - p[star], axis=-1)
            hit = idx[d < r_star]
            n_star += len(hit)
            alive[hit] = False
            mass[star] += mass[hit].sum()
            idx = np.nonzero(alive)[0]
        q = torch.from_numpy(p[idx].astype(np.float64))
        dm = torch.cdist(q, q)
        dm.fill_diagonal_(1e9)
        ii, jj = torch.nonzero(torch.triu(dm < r_merge), as_tuple=True)
        for i, j in zip(idx[ii.numpy()], idx[jj.numpy()]):
            if alive[i] and alive[j]:
                big, small = (i, j) if mass[i] >= mass[j] else (j, i)
                mass[big] += mass[small]
                alive[small] = False
                n_merge += 1
    return alive, mass, n_merge, n_star


def run_real(args):
    rng = np.random.default_rng(args.seed)
    jup = [5.0] if args.jupiters == 1 else [3.0, 4.0, 5.0]
    pos, vel, mass = real_ic(rng, args.n_disk, jup)
    nj = len(jup)
    dev = torch.device(args.device)
    p, v, m = (torch.tensor(x, dtype=torch.float64, device=dev) for x in (pos, vel, mass))
    h = args.dt
    nsteps = int(round(args.years / h))
    t0 = time.time()
    sp, sv, ticks, diag, (e0, l0) = leapfrog(p, v, m, args.eps, G_REAL, h, nsteps, args.every, log=True)
    print("wall", time.time() - t0, flush=True)
    a0, ecc0 = helio_elements(pos[None], vel[None], mass, G_REAL)
    a1, ecc1 = helio_elements(sp[-1:].astype(np.float64), sv[-1:].astype(np.float64), mass, G_REAL)
    a0, ecc0, a1, ecc1 = a0[0], ecc0[0], a1[0], ecc1[0]
    di = np.arange(1 + nj, len(mass))
    sel = (a0[nj:] > 0.9) & (a0[nj:] < 1.1)
    bound = a1[nj:] > 0
    da = np.abs(a1[nj:] - a0[nj:]) / a0[nj:]
    out = dict(mode="real", n_disk=args.n_disk, jupiters=jup, years=args.years, dt=h, eps=args.eps, seed=args.seed,
               dE_over_E0=diag[-1][1], dL_over_L0=diag[-1][2], max_abs_dE=max(abs(d[1]) for d in diag),
               median_da_over_a_disk=float(np.median(da[bound])), median_ecc_end=float(np.median(ecc1[nj:][bound])),
               median_ecc_start=float(np.median(ecc0[nj:])), median_da_over_a_1AU=float(np.median(da[sel & bound])) if (sel & bound).any() else None,
               unbound_disk_end=int((~bound).sum()))
    for k, j in enumerate(jup):
        out[f"jupiter_{k}_a_start_end"] = [float(a0[k]), float(a1[k])]
        out[f"jupiter_{k}_ecc_end"] = float(ecc1[k])
    gap0 = ((np.abs(a0[nj:] - 5.0) < 0.7)).sum()
    gap1 = ((np.abs(a1[nj:] - 5.0) < 0.7) & bound).sum()
    out["gap_fraction_remaining_within_0.7AU_of_5AU"] = float(gap1 / max(gap0, 1))
    alive, m2, n_merge, n_star = merge_count(sp[:, :, :], mass, np.arange(1, len(mass)), args.r_merge, star=0, r_star=0.02)
    out.update(merges=n_merge, accreted_onto_star=n_star, survivors=int(alive.sum()), survivor_mass_disk_plus_jup=float(m2[alive].sum()),
               snapshot_every_yr=args.every * h, r_merge_AU=args.r_merge)
    print(json.dumps(out, indent=1))
    json.dump(out, open(args.out, "w"), indent=1)
    np.savez_compressed(args.out.replace(".json", ".npz"), pos=sp[::10], mass=mass, ticks=ticks[::10])


def proxy_ic(rng, n_core=300, n_disk=700, a_core=15.0, r_in=20.0, r_out=55.0, g=1.0, eps=0.5):
    u = rng.uniform(size=n_core)
    r3 = a_core / np.sqrt(u ** (-2 / 3) - 1)
    d3 = rng.normal(size=(n_core, 3))
    d3 /= np.linalg.norm(d3, axis=1, keepdims=True)
    cpos = (r3[:, None] * d3)[:, :2]
    sig = np.sqrt(g * n_core / (6 * np.sqrt(r3 ** 2 + a_core ** 2)))
    cvel = rng.normal(size=(n_core, 2)) * sig[:, None]
    uu = rng.uniform(size=n_disk)
    r = (r_in ** -0.5 - uu * (r_in ** -0.5 - r_out ** -0.5)) ** -2
    th = rng.uniform(0, 2 * np.pi, n_disk)
    dpos = np.stack([r * np.cos(th), r * np.sin(th)], -1)
    pos = np.concatenate([cpos, dpos])
    vel = np.concatenate([cvel, np.zeros((n_disk, 2))])
    t = torch.tensor(pos)
    acc = accel(t, torch.ones(len(pos), dtype=torch.float64), eps, g).numpy()[n_core:]
    rhat = dpos / r[:, None]
    ar = -(acc * rhat).sum(-1)
    vc = np.sqrt(np.clip(r * ar, 1e-6, None))
    that = np.stack([-np.sin(th), np.cos(th)], -1)
    vd = vc[:, None] * that * (1 + 0.01 * rng.normal(size=(n_disk, 1))) + 0.01 * vc[:, None] * rng.normal(size=(n_disk, 2))
    vel[n_core:] = vd
    pos -= pos.mean(0)
    vel -= vel.mean(0)
    return pos, vel


def proxy_elements(snaps, lo, hi, com):
    r = np.linalg.norm(snaps[lo:hi] - com[lo:hi, None, :], axis=-1)
    rp, ra = r.min(0), r.max(0)
    return (rp + ra) / 2, (ra - rp) / (ra + rp)


def run_proxy(args):
    sys.path.insert(0, args.repo)
    from model.central_force import CentralForceDynamics
    rng = np.random.default_rng(args.seed)
    pos, vel = proxy_ic(rng, args.n_core, args.n_disk)
    n = len(pos)
    dev = torch.device(args.device)
    mass = torch.ones(n, dtype=torch.float64, device=dev)
    ticks_total = args.ticks
    sub = 4
    h = 0.1 / sub
    p, v = torch.tensor(pos, device=dev), torch.tensor(vel, device=dev)
    t0 = time.time()
    rp_, rv_, rt, diag, (e0, l0) = leapfrog(p, v, mass, 0.5, 1.0, h, ticks_total * sub, 10 * sub, first_dense=0, log=True)
    # dense first 100 ticks for err@k, recomputed cheaply
    dp, dv, dt_, _, _ = leapfrog(p, v, mass, 0.5, 1.0, h, 100 * sub, sub)
    print("ref wall", time.time() - t0, flush=True)
    dyn = CentralForceDynamics(dt=0.1, neighbor_radius=100.0).to(dev)
    dyn.load_state_dict(torch.load(args.ckpt, map_location=dev))
    mp, mv = p.float().clone(), v.float().clone()
    hid = torch.zeros(n, 1, device=dev)
    mdp, mdiag, msnap = [mp.cpu().numpy()], [], []
    mdense = [mp.cpu().numpy()]
    me0, ml0 = energy(mp.double(), mv.double(), mass, 0.5, 1.0), ang_mom(mp.double(), mv.double(), mass)
    t0 = time.time()
    finite = True
    with torch.no_grad():
        for s in range(1, ticks_total + 1):
            dpp, dvv, hid = dyn(mp, mv, hid)
            mp = mp + mv * 0.1 + dpp
            mv = mv + dvv
            if s <= 100:
                mdense.append(mp.cpu().numpy())
            if s % 10 == 0:
                msnap.append(mp.cpu().numpy())
            if s % (ticks_total // 20) == 0:
                finite = bool(torch.isfinite(mp).all())
                de = (energy(mp.double(), mv.double(), mass, 0.5, 1.0) - me0) / abs(me0)
                dl = float((ang_mom(mp.double(), mv.double(), mass) - ml0).abs() / abs(ml0))
                mdiag.append((s, de, dl, finite))
                print(f"model tick {s} dE/E0 {de:.3e} dL/L0 {dl:.3e} finite {finite}", flush=True)
    print("model wall", time.time() - t0, flush=True)
    mdense, msnap = np.stack(mdense), np.stack(msnap)
    ref_dense = np.concatenate([[pos.astype(np.float32)], dp])
    err = {k: float(np.linalg.norm(mdense[k] - ref_dense[k], axis=-1).mean()) for k in (1, 5, 20, 50, 100)}
    ncore = args.n_core
    nsn = len(msnap)
    w = max(nsn // 10, 2)
    refc = rp_.mean(1)
    modc = msnap.mean(1)
    res = {}
    for name, S, com in (("ref", rp_[:, ncore:] * 0 + rp_[:, ncore:], refc), ("model", msnap[:, ncore:], modc)):
        S2 = rp_ if name == "ref" else msnap
        a0_, e0_ = proxy_elements(S2[:, ncore:], 0, w, com)
        a1_, e1_ = proxy_elements(S2[:, ncore:], nsn - w, nsn, com)
        fin = np.isfinite(S2[-1]).all(1)
        r_end = np.linalg.norm(S2[-1][ncore:] - com[-1], axis=-1)
        surv = float((r_end < 150).mean())
        alive, m2, nm, _ = merge_count(S2[:, ncore:].astype(np.float64), np.ones(args.n_disk), np.arange(args.n_disk), args.r_merge)
        res[name] = dict(median_abs_da_over_a=float(np.nanmedian(np.abs(a1_ - a0_) / a0_)), median_e_start=float(np.nanmedian(e0_)),
                         median_e_end=float(np.nanmedian(e1_)), survivor_fraction=surv, merges=nm, survivors_after_merge=int(alive.sum()))
    out = dict(mode="proxy", n=n, n_core=ncore, n_disk=args.n_disk, ticks=ticks_total, seed=args.seed, err_at_k=err,
               ref_diag=[(int(s) // 4, d, l) for s, d, l in diag], model_diag=[(int(s), float(d), l, f) for s, d, l, f in mdiag],
               ref=res["ref"], model=res["model"], r_merge=args.r_merge)
    print(json.dumps(out, indent=1, default=float))
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    np.savez_compressed(args.out.replace(".json", ".npz"), ref=rp_[::5].astype(np.float32), model=msnap[::5].astype(np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "proxy"])
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-disk", type=int, default=1000)
    ap.add_argument("--n-core", type=int, default=300)
    ap.add_argument("--jupiters", type=int, default=1)
    ap.add_argument("--years", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=1 / 640)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--r-merge", type=float, default=0.02)
    ap.add_argument("--ckpt", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--ticks", type=int, default=15000)
    a = ap.parse_args()
    run_real(a) if a.mode == "real" else run_proxy(a)
