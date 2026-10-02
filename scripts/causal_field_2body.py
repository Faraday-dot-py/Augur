"""Causal-field 2-body prototype. Star (mass 1) + planet (mass m), softened 1/r^2 gravity (G=1, eps=0.5), KDK leapfrog.
Variants: instant (analytic Newton), retarded (Newton force from the source's retarded position/time, history buffer),
token (learned CentralForce pair force, instantaneous), field (3D scalar wave equation phi_tt = c^2 (lap phi - 4 pi rho) on a grid, one field per
source body so no self-force, particles read -grad phi at the orbit plane).
Sections: delete_truth delete_token delete_field cfl orbit_truth orbit_token orbit_field; `plot` renders from results (local).
Usage: PYTHONPATH=. python scripts/causal_field_2body.py run --sections all [--quick]
"""
import argparse
import json
import math
import time

import numpy as np

EPS = 0.5
SEED = 4738
C = 3.0
H = 0.05
T0 = 30.0
OUT = "results/causal_field_2body"


def circ_speed(mtot, R, eps=EPS):
    return math.sqrt(mtot * R * R / (R * R + eps * eps) ** 1.5)


def onset(t, a, t0, w=10.0):
    t, a = np.asarray(t), np.asarray(a)
    pre = (t >= t0 - w) & (t < t0)
    a0 = a[pre].mean()
    post = t >= t0
    tp, ap = t[post], a[post]

    def first(mask):
        i = np.flatnonzero(mask)
        return float(tp[i[0]] - t0) if len(i) else float("nan")
    return {"a0": float(a0), "pre_std_rel": float(a[pre].std() / a0), "lat50": first(ap < 0.5 * a0),
            "lat05": first(np.abs(ap - a0) > 0.05 * a0), "t90": first(ap < 0.9 * a0), "t10": first(ap < 0.1 * a0),
            "peak_rel": float(ap.max() / a0), "t_peak": float(tp[ap.argmax()] - t0), "min_rel": float(ap.min() / a0)}


def orbit_stats(t, pos, vel, mass, R, eps=EPS):
    pos, vel = np.asarray(pos), np.asarray(vel)
    mass = [mass[0], mass[1] if mass[1] > 0 else 1.0]
    rel = pos[:, 1] - pos[:, 0]
    r = np.sqrt((rel ** 2).sum(-1))
    ke = 0.5 * (mass[0] * (vel[:, 0] ** 2).sum(-1) + mass[1] * (vel[:, 1] ** 2).sum(-1))
    e = ke - mass[0] * mass[1] / np.sqrt(r ** 2 + eps ** 2)
    ang = (mass[0] * (pos[:, 0, 0] * vel[:, 0, 1] - pos[:, 0, 1] * vel[:, 0, 0]) + mass[1] * (pos[:, 1, 0] * vel[:, 1, 1] - pos[:, 1, 1] * vel[:, 1, 0]))
    mom = np.linalg.norm(mass[0] * vel[:, 0] + mass[1] * vel[:, 1], axis=-1)
    return {"t": t, "r": r, "E": e, "L": ang, "P": mom}


def orbit_summary(s, orbit_period):
    e0, l0 = s["E"][0], s["L"][0]
    return {"dE_rel_final": float((s["E"][-1] - e0) / abs(e0)), "dE_rel_maxabs": float(np.abs((s["E"] - e0) / abs(e0)).max()),
            "dL_rel_final": float((s["L"][-1] - l0) / abs(l0)), "r_min": float(s["r"].min()), "r_max": float(s["r"].max()),
            "r_final": float(s["r"][-1]), "P_max": float(s["P"].max()), "orbits": float(s["t"][-1] / orbit_period),
            "dE_rel_per_orbit": float((s["E"][-1] - e0) / abs(e0) / (s["t"][-1] / orbit_period)), "finite": bool(np.isfinite(s["E"]).all())}


def init_pair(R, m, M=1.0):
    vrel = circ_speed(M + m, R)
    pos = [[-m / (M + m) * R, 0.0], [M / (M + m) * R, 0.0]]
    vel = [[0.0, -m / (M + m) * vrel], [0.0, M / (M + m) * vrel]]
    return pos, vel


def run_truth(torch, dev, R, m, c, steps, n0, retarded, h=H, M=1.0, eps=EPS):
    B = len(R)
    f64 = torch.float64
    R_, m_, c_ = (torch.tensor(x, dtype=f64, device=dev) for x in (R, m, c))
    pos = torch.zeros(B, 2, 2, dtype=f64, device=dev)
    vel = torch.zeros_like(pos)
    for b in range(B):
        p, v = init_pair(R[b], m[b], M)
        pos[b], vel[b] = torch.tensor(p, dtype=f64, device=dev), torch.tensor(v, dtype=f64, device=dev)
    mass = torch.stack([torch.full_like(m_, M), m_], 1)
    bi = torch.arange(B, device=dev)[:, None].expand(B, 2)
    ji = torch.tensor([1, 0], device=dev)[None].expand(B, 2)
    hist = torch.zeros(steps + 1, B, 2, 2, dtype=f64, device=dev)
    vh = torch.zeros_like(hist)
    ah = torch.zeros(steps + 1, B, 2, dtype=f64, device=dev)
    msrc = mass[bi, ji]
    mind = [float("inf")]

    def force(pos, t):
        if retarded:
            nv = max(t - 1, 0)
            s = torch.full((B, 2), float(nv), dtype=f64, device=dev)

            def src_at(s):
                s = s.clamp(0, nv)
                f = s.floor().long().clamp(max=max(nv - 1, 0))
                w = (s - f)[..., None]
                return hist[f, bi, ji] * (1 - w) + hist[(f + 1).clamp(max=nv), bi, ji] * w
            for _ in range(8):
                d = (pos - src_at(s)).norm(dim=-1)
                s = t - d / (c_[:, None] * h)
            src = src_at(s)
            alive = torch.where(ji == 0, s <= n0, torch.ones_like(s, dtype=torch.bool)) if n0 is not None else torch.ones_like(s, dtype=torch.bool)
        else:
            src = pos[bi, ji]
            alive = torch.ones((B, 2), dtype=torch.bool, device=dev) if n0 is None else torch.where(ji == 0, torch.tensor(t <= n0, device=dev), torch.ones((B, 2), dtype=torch.bool, device=dev))
        rel = src - pos
        d2 = (rel ** 2).sum(-1)
        return (msrc * alive)[..., None] * rel / ((d2 + eps ** 2) ** 1.5)[..., None], d2.sqrt()

    hist[0], vh[0] = pos, vel
    a, _ = force(pos, 0)
    ah[0] = a[:, 1]
    for n in range(steps):
        vel = vel + 0.5 * h * a
        pos = pos + h * vel
        hist[n + 1] = pos
        a, d = force(pos, n + 1)
        vel = vel + 0.5 * h * a
        vh[n + 1] = vel
        ah[n + 1] = a[:, 1]
    mass_c = mass.cpu().numpy()
    return {"t": np.arange(steps + 1) * h, "pos": hist.cpu().numpy(), "vel": vh.cpu().numpy(), "acc_p": ah.cpu().numpy(), "mass": mass_c}


def run_field(torch, dev, R, m, c, steps, n0, nsub, dx, L, h=H, M=1.0, eps=EPS, rec_every=1, warm_tc=4.0):
    f64 = torch.float64
    dtf = h / nsub
    kap = c * dtf / dx
    k2 = kap ** 2
    N = int(round(2 * L / dx)) + 1
    c0 = (N - 1) // 2
    zc = c0
    masses = [M, m]
    act = [j for j in (0, 1) if masses[j] > 0]
    fidx = {j: k for k, j in enumerate(act)}
    nb = len(act)
    ax = (torch.arange(N, device=dev, dtype=f64) - c0) * dx
    X, Y, Z = ax[:, None, None], ax[None, :, None], ax[None, None, :]
    pp, vv = init_pair(R, m, M)
    pos = torch.tensor(pp, dtype=f64, device=dev)
    vel = torch.tensor(vv, dtype=f64, device=dev)
    phi = torch.zeros(nb, N, N, N, dtype=f64, device=dev)
    for j in act:
        phi[fidx[j]] = -masses[j] / ((X - pos[j, 0]) ** 2 + (Y - pos[j, 1]) ** 2 + Z ** 2 + (0.5 * dx) ** 2).sqrt()
    old = phi.clone()
    mask = torch.zeros(N, N, N, dtype=torch.bool, device=dev)
    mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1] = True, True, True, True, True, True
    bidx = mask.flatten().nonzero()[:, 0]
    bx, by, bz = X.expand(N, N, N).flatten()[bidx], Y.expand(N, N, N).flatten()[bidx], Z.expand(N, N, N).flatten()[bidx]
    rb = (bx ** 2 + by ** 2 + bz ** 2).sqrt()
    del mask, X, Y, Z
    t0 = None if n0 is None else n0 * h
    four_pi = 4 * math.pi
    state = {"phi": phi, "old": old, "t": -warm_tc * L / c}
    dtype_idx = torch.long

    def nodes(p):
        g = p / dx + c0
        ix = g.floor().to(dtype_idx).clamp(1, N - 3)
        f = (g - ix).clamp(0, 1)
        ixn = torch.stack([ix[0], ix[0] + 1, ix[0], ix[0] + 1])
        iyn = torch.stack([ix[1], ix[1], ix[1] + 1, ix[1] + 1])
        w = torch.stack([(1 - f[0]) * (1 - f[1]), f[0] * (1 - f[1]), (1 - f[0]) * f[1], f[0] * f[1]])
        return ixn, iyn, w

    def fstep(src_pos, damp=0.0):
        phi, old, t = state["phi"], state["old"], state["t"]
        new = phi + (1 - damp) * (phi - old)
        lap = (phi[:, 2:, 1:-1, 1:-1] + phi[:, :-2, 1:-1, 1:-1] + phi[:, 1:-1, 2:, 1:-1] + phi[:, 1:-1, :-2, 1:-1]
               + phi[:, 1:-1, 1:-1, 2:] + phi[:, 1:-1, 1:-1, :-2] - 6 * phi[:, 1:-1, 1:-1, 1:-1])
        new[:, 1:-1, 1:-1, 1:-1] += k2 * lap
        for j in act:
            if j == 0 and t0 is not None and t > t0:
                continue
            ixn, iyn, w = nodes(src_pos[j])
            zi = torch.full_like(ixn, zc)
            new[fidx[j]].index_put_((ixn, iyn, zi), -k2 * four_pi * masses[j] * w / dx, accumulate=True)
        flat = new.view(nb, -1)
        for j in act:
            val = -masses[j] / rb.clamp(min=1e-9)
            if j == 0 and t0 is not None:
                val = val * ((t + dtf - rb / c) <= t0)
            flat[fidx[j]].index_copy_(0, bidx, val)
        state["old"], state["phi"], state["t"] = phi, new, t + dtf

    def read(pos):
        a = torch.zeros(2, 2, dtype=f64, device=dev)
        out = []
        for i in (0, 1):
            g = torch.zeros(2, dtype=f64, device=dev)
            for j in act:
                if j == i:
                    continue
                ixn, iyn, w = nodes(pos[i])
                plane = state["phi"][fidx[j], :, :, zc]
                gx = (plane[ixn + 1, iyn] - plane[ixn - 1, iyn]) / (2 * dx)
                gy = (plane[ixn, iyn + 1] - plane[ixn, iyn - 1]) / (2 * dx)
                g = g + torch.stack([(w * gx).sum(), (w * gy).sum()])
            out.append(-g)
        return torch.stack(out)

    nwarm = int(round(warm_tc * L / c / dtf))
    for _ in range(nwarm):
        fstep(pos, damp=0.03)
    state["t"] = 0.0
    T = steps // rec_every + 1
    ph = torch.zeros(T, 2, 2, dtype=f64, device=dev)
    vh = torch.zeros_like(ph)
    ah = torch.zeros(T, 2, dtype=f64, device=dev)
    a = read(pos)
    ph[0], vh[0], ah[0] = pos, vel, a[1]
    maxphi = []
    blew = False
    for n in range(steps):
        vel = vel + 0.5 * h * a
        newpos = pos + h * vel
        for k in range(nsub):
            fstep(pos + (newpos - pos) * (k / nsub))
        pos = newpos
        a = read(pos)
        vel = vel + 0.5 * h * a
        if (n + 1) % rec_every == 0:
            i = (n + 1) // rec_every
            ph[i], vh[i], ah[i] = pos, vel, a[1]
        if (n + 1) % 50 == 0:
            mx = float(state["phi"].abs().max())
            maxphi.append(mx)
            if not math.isfinite(mx) or mx > 1e8:
                blew = True
                steps_done = (n + 1) // rec_every + 1
                ph, vh, ah = ph[:steps_done], vh[:steps_done], ah[:steps_done]
                break
    return {"t": np.arange(len(ph)) * h * rec_every, "pos": ph.cpu().numpy(), "vel": vh.cpu().numpy(), "acc_p": ah.cpu().numpy(),
            "mass": np.array(masses), "kappa": kap, "N": N, "blew": blew, "maxphi": maxphi}


def run_token(torch, dev, R, steps, n0, h=H, ckpt="checkpoints/gravity_central_v1.pt", orbit=False):
    from model.central_force import CentralForceDynamics
    dyn = CentralForceDynamics(dt=0.1).to(dev)
    dyn.load_state_dict(torch.load(ckpt, map_location=dev))
    dyn.eval()
    star = torch.zeros(1, 2, device=dev)

    def acc(p, alive=True):
        if not alive:
            return torch.zeros(2, device=dev)
        with torch.no_grad():
            return dyn.accel(torch.cat([star, p[None]]))[1]
    a_an = R / (R * R + EPS ** 2) ** 1.5
    p = torch.tensor([R, 0.0], device=dev)
    a_l = float(acc(p).norm())
    v = torch.tensor([0.0, math.sqrt(R * a_l)], device=dev)
    pos_l, vel_l, acc_l = [p.cpu().numpy()], [v.cpu().numpy()], []
    a = acc(p)
    acc_l.append(a.cpu().numpy())
    for n in range(steps):
        v = v + 0.5 * h * a
        p = p + h * v
        a = acc(p, alive=(n + 1) <= n0 if n0 is not None else True)
        v = v + 0.5 * h * a
        pos_l.append(p.cpu().numpy())
        vel_l.append(v.cpu().numpy())
        acc_l.append(a.cpu().numpy())
    return {"t": np.arange(steps + 1) * h, "pos": np.array(pos_l), "vel": np.array(vel_l), "acc_p": np.array(acc_l), "a_ratio": a_l / a_an}


def mag(x):
    return np.linalg.norm(x, axis=-1)


def sec_delete_truth(torch, dev, res, ts, quick):
    Rs = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0]
    cs = [1.5, 3.0, 6.0, 12.0]
    cfgs = [(R, C) for R in Rs] + [(10.0, c) for c in cs if c != C]
    n0 = int(round(T0 / H))
    steps = int(round((T0 + 2.5 * max(r / c for r, c in cfgs) + 10) / H))
    out = {}
    for name, ret in (("instant", False), ("retarded", True)):
        r = run_truth(torch, dev, [x[0] for x in cfgs], [0.0] * len(cfgs), [x[1] for x in cfgs], steps, n0, ret)
        for b, (R, c) in enumerate(cfgs):
            o = onset(r["t"], mag(r["acc_p"][:, b]), T0)
            o["pred_lat"] = 0.0 if not ret else R / c
            out[f"{name}_R{R}_c{c}"] = o
        b10 = cfgs.index((10.0, C))
        ts[f"delete_{name}_R10_t"], ts[f"delete_{name}_R10_a"] = r["t"], mag(r["acc_p"][:, b10])
        print(name, {k: round(v["lat50"], 3) for k, v in out.items() if k.startswith(name)}, flush=True)
    res["delete_truth"] = out


def sec_delete_token(torch, dev, res, ts, quick):
    out = {}
    n0 = int(round(T0 / H))
    for R in (5.0, 10.0, 20.0, 30.0):
        steps = n0 + int(round((2.5 * R / C + 10) / H))
        r = run_token(torch, dev, R, steps, n0)
        o = onset(r["t"], mag(r["acc_p"]), T0)
        o["a_ratio_vs_analytic"] = r["a_ratio"]
        out[f"token_R{R}_c{C}"] = o
        if R == 10.0:
            ts["delete_token_R10_t"], ts["delete_token_R10_a"] = r["t"], mag(r["acc_p"])
    res["delete_token"] = out
    print("token", {k: round(v["lat50"], 3) for k, v in out.items()}, flush=True)


def sec_delete_field(torch, dev, res, ts, quick):
    out = {}
    n0 = int(round(T0 / H))
    cases = [(R, C) for R in (5.0, 10.0, 20.0, 30.0)] + [(10.0, 1.5), (10.0, 6.0)]
    if quick:
        cases = [(10.0, C)]
    for R, c in cases:
        nsub = max(1, math.ceil(c * H / (0.3 * 0.5)))
        steps = n0 + int(round((2.5 * R / c + 10) / H))
        L = math.ceil(R + 28.0)
        t1 = time.time()
        r = run_field(torch, dev, R, 0.0, c, steps, n0, nsub, 0.5, L)
        o = onset(r["t"], mag(r["acc_p"]), T0)
        o["pred_lat"] = R / c
        o["kappa"], o["nsub"], o["N"], o["wall_s"] = r["kappa"], nsub, r["N"], time.time() - t1
        out[f"field_R{R}_c{c}"] = o
        if (R, c) == (10.0, C):
            ts["delete_field_R10_t"], ts["delete_field_R10_a"] = r["t"], mag(r["acc_p"])
        print("field", R, c, {k: round(v, 3) if isinstance(v, float) else v for k, v in o.items()}, flush=True)
    res["delete_field"] = out


def sec_cfl(torch, dev, res, ts, quick):
    out = {}
    R, c, dx = 10.0, C, 0.5
    kaps = [0.1, 0.3, 0.5, 0.55, 0.6, 0.7]
    for kap in kaps:
        h = kap * dx / c
        n0 = int(round(T0 / h))
        steps = n0 + int(round((2.5 * R / c + 10) / h))
        r = run_field(torch, dev, R, 0.0, c, steps, n0, 1, dx, 32.0, h=h, rec_every=max(1, int(round(0.05 / h))))
        o = {"kappa": kap, "h": h, "blew": r["blew"], "maxphi_last": r["maxphi"][-1] if r["maxphi"] else None}
        if not r["blew"]:
            o.update(onset(r["t"], mag(r["acc_p"]), T0))
        o["pred_lat"] = R / c
        out[f"kappa{kap}"] = o
        ts[f"cfl_k{kap}_t"], ts[f"cfl_k{kap}_a"] = r["t"], mag(r["acc_p"])
        print("cfl", kap, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    for dx in (1.0, 0.5, 0.25):
        h = H
        nsub = max(1, math.ceil(c * h / (0.3 * dx)))
        n0 = int(round(T0 / h))
        steps = n0 + int(round((2.5 * R / c + 10) / h))
        r = run_field(torch, dev, R, 0.0, c, steps, n0, nsub, dx, 32.0)
        o = onset(r["t"], mag(r["acc_p"]), T0)
        o.update({"dx": dx, "kappa": r["kappa"], "nsub": nsub, "N": r["N"], "pred_lat": R / c, "a_static_ratio": o["a0"] / (R / (R * R + EPS ** 2) ** 1.5)})
        out[f"dx{dx}"] = o
        ts[f"cfl_dx{dx}_t"], ts[f"cfl_dx{dx}_a"] = r["t"], mag(r["acc_p"])
        print("dx", dx, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    res["cfl"] = out


def orbit_run_summary(r, R, m, per, tag, ts, every=100):
    mass = r["mass"] if r["mass"].ndim == 1 else r["mass"][0]
    s = orbit_stats(r["t"], r["pos"][:, 0] if r["pos"].ndim == 4 else r["pos"], r["vel"][:, 0] if r["vel"].ndim == 4 else r["vel"], mass, R)
    o = orbit_summary(s, per)
    for k in ("t", "r", "E", "L"):
        ts[f"orbit_{tag}_{k}"] = s[k][::every]
    return o


def sec_orbit_truth(torch, dev, res, ts, quick):
    R = 10.0
    cfgs = [(0.0, C), (0.01, C), (0.1, C), (1.0, C), (0.1, 1.5), (1.0, 1.5)]
    norb = 2 if quick else 10
    per = 2 * math.pi * R / circ_speed(1.0, R)
    steps = int(round(norb * per / H))
    out = {}
    for name, ret in (("instant", False), ("retarded", True)):
        t1 = time.time()
        r = run_truth(torch, dev, [R] * len(cfgs), [x[0] for x in cfgs], [x[1] for x in cfgs], steps, None, ret)
        for b, (m, c) in enumerate(cfgs):
            rb = {"t": r["t"], "pos": r["pos"][:, b], "vel": r["vel"][:, b], "mass": r["mass"][b]}
            pm = 2 * math.pi * R / circ_speed(1.0 + m, R)
            o = orbit_run_summary(rb, R, m, pm, f"{name}_m{m}_c{c}", ts)
            out[f"{name}_m{m}_c{c}"] = o
            print(name, m, c, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
        print("wall", name, time.time() - t1, flush=True)
    res["orbit_truth"] = out


def sec_orbit_token(torch, dev, res, ts, quick):
    R = 10.0
    norb = 2 if quick else 10
    per = 2 * math.pi * R / circ_speed(1.0, R)
    steps = int(round(norb * per / H))
    r = run_token(torch, dev, R, steps, None)
    rel = r["pos"]
    rr = np.linalg.norm(rel, axis=-1)
    ang = rel[:, 0] * r["vel"][:, 1] - rel[:, 1] * r["vel"][:, 0]
    res["orbit_token"] = {"a_ratio_vs_analytic": r["a_ratio"], "r_min": float(rr.min()), "r_max": float(rr.max()), "dL_rel_final": float((ang[-1] - ang[0]) / ang[0]),
                          "orbits": norb, "note": "test particle in learned central force, star fixed; E not computed (no learned potential here)"}
    ts["orbit_token_t"], ts["orbit_token_r"] = r["t"][::100], rr[::100]
    print("orbit_token", res["orbit_token"], flush=True)


def sec_orbit_field(torch, dev, res, ts, quick):
    R, dx = 10.0, 0.5
    norb = 2 if quick else 10
    out = {}
    for m, c in [(0.0, C), (0.1, C), (1.0, C)] if not quick else [(0.0, C)]:
        per = 2 * math.pi * R / circ_speed(1.0 + m, R)
        steps = int(round(norb * per / H))
        nsub = max(1, math.ceil(c * H / (0.3 * dx)))
        t1 = time.time()
        r = run_field(torch, dev, R, m, c, steps, None, nsub, dx, 32.0, rec_every=1)
        mass = r["mass"]
        s = orbit_stats(r["t"], r["pos"], r["vel"], mass, R)
        o = orbit_summary(s, per)
        o.update({"blew": r["blew"], "kappa": r["kappa"], "wall_s": time.time() - t1})
        for k in ("t", "r", "E", "L"):
            ts[f"orbit_field_m{m}_c{c}_{k}"] = s[k][::100]
        out[f"field_m{m}_c{c}"] = o
        print("orbit_field", m, c, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    res["orbit_field"] = out


SECTIONS = {"delete_truth": sec_delete_truth, "delete_token": sec_delete_token, "delete_field": sec_delete_field, "cfl": sec_cfl,
            "orbit_truth": sec_orbit_truth, "orbit_token": sec_orbit_token, "orbit_field": sec_orbit_field}


def run(args):
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    names = list(SECTIONS) if args.sections == "all" else args.sections.split(",")
    res, ts = {}, {}
    tag = args.tag
    for nm in names:
        t1 = time.time()
        SECTIONS[nm](torch, dev, res, ts, args.quick)
        res[nm + "_wall_s"] = time.time() - t1
        print("section", nm, "done", time.time() - t1, flush=True)
        json.dump({"args": vars(args), "c": C, "h": H, "eps": EPS, "T0": T0, "results": res}, open(f"{OUT}{tag}.json", "w"), indent=1)
        np.savez_compressed(f"{OUT}{tag}_ts.npz", **ts)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = args.tag
    j = json.load(open(f"{OUT}{tag}.json"))["results"]
    ts = np.load(f"{OUT}{tag}_ts.npz")
    col = {"instant": "#1f77b4", "retarded": "#d62728", "token": "#2ca02c", "field": "#ff7f0e"}
    R = 10.0
    tau = R / C
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in ("instant", "retarded", "token", "field"):
        k = f"delete_{v}_R10_t"
        if k in ts:
            ax.plot((ts[k] - T0) / tau, ts[f"delete_{v}_R10_a"] / ts[f"delete_{v}_R10_a"][(ts[k] > T0 - 10) & (ts[k] < T0)].mean(), label=v, color=col[v], lw=1.6)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.axvline(1, color="k", ls="--", lw=0.8, label="t0 + R/c")
    ax.set_xlim(-3, 4)
    ax.set_xlabel("(t - t0) / (R/c)")
    ax.set_ylabel("|planet accel| / pre-deletion value")
    ax.set_title("Star deleted at t0: planet acceleration (R=10, c=3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("videos/causal_field_2body_accel.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for v, d in (("instant", j["delete_truth"]), ("retarded", j["delete_truth"]), ("token", j.get("delete_token", {})), ("field", j.get("delete_field", {}))):
        pts = sorted((float(k.split("_R")[1].split("_c")[0]), val["lat50"]) for k, val in d.items() if k.startswith(v) and k.endswith(f"_c{C}"))
        if pts:
            axs[0].plot(*zip(*pts), "o-", label=v, color=col[v])
        pts = sorted((float(k.split("_c")[1]), val["lat50"]) for k, val in d.items() if k.startswith(v) and "_R10.0_" in k)
        if len(pts) > 1:
            axs[1].plot(*zip(*pts), "o-", label=v, color=col[v])
    rr = np.linspace(0, 40, 50)
    axs[0].plot(rr, rr / C, "k--", lw=0.8, label="R/c")
    axs[0].set_xlabel("R (star-planet distance)")
    axs[0].set_ylabel("reaction latency (50% drop)")
    axs[0].legend()
    cc = np.linspace(1, 12, 50)
    axs[1].plot(cc, 10.0 / cc, "k--", lw=0.8, label="R/c")
    axs[1].set_xlabel("c (R=10)")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig("videos/causal_field_2body_latency.png", dpi=130)
    plt.close(fig)

    keys = sorted({k[len("orbit_"):-2] for k in ts.files if k.startswith("orbit_") and k.endswith("_E")})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in keys:
        if "token" in k:
            continue
        t, e, r = ts[f"orbit_{k}_t"], ts[f"orbit_{k}_E"], ts[f"orbit_{k}_r"]
        v = k.split("_")[0]
        m = k.split("_m")[1].split("_")[0]
        c = k.split("_c")[1] if "_c" in k else str(C)
        if m not in ("0.0", "1.0") or c not in ("3.0",):
            continue
        per = 2 * math.pi * R / circ_speed(1.0 + float(m), R)
        axs[0].plot(t / per, (e - e[0]) / abs(e[0]), label=f"{v} m={m}", color=col[v], ls="-" if m == "1.0" else ":")
        axs[1].plot(t / per, r, label=f"{v} m={m}", color=col[v], ls="-" if m == "1.0" else ":")
    axs[0].set_xlabel("orbits")
    axs[0].set_ylabel("(E - E0) / |E0|")
    axs[0].set_yscale("symlog", linthresh=1e-6)
    axs[1].set_xlabel("orbits")
    axs[1].set_ylabel("separation")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("videos/causal_field_2body_orbit.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in sorted(k for k in ts.files if k.startswith("cfl_k") and k.endswith("_t")):
        kap = k[len("cfl_k"):-2]
        t, a = ts[k], ts[k.replace("_t", "_a")]
        pre = (t > T0 - 10) & (t < T0)
        axs[0].plot((t - T0) / tau, a / a[pre].mean(), label=f"kappa {kap}")
    axs[0].set_ylim(-0.5, 3)
    axs[0].set_xlim(-3, 4)
    axs[0].set_xlabel("(t - t0) / (R/c)")
    axs[0].set_ylabel("|a| / pre")
    axs[0].legend(fontsize=7)
    for k in sorted(k for k in ts.files if k.startswith("cfl_dx") and k.endswith("_t")):
        t, a = ts[k], ts[k.replace("_t", "_a")]
        pre = (t > T0 - 10) & (t < T0)
        axs[1].plot((t - T0) / tau, a / a[pre].mean(), label=f"dx {k[len('cfl_dx'):-2]}")
    axs[1].set_ylim(-0.5, 3)
    axs[1].set_xlim(-3, 4)
    axs[1].set_xlabel("(t - t0) / (R/c)")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("videos/causal_field_2body_cfl.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "plot"])
    ap.add_argument("--sections", default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else plot(args)


if __name__ == "__main__":
    main()
