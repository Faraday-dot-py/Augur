"""Scene generalization of the scatter-field model: bounce, black hole, orbit, globular cluster; zero-shot / fine-tune / full-train tiers.
See docs/debugging/scatter-scenes.md for criteria and predictions.

  python scripts/scatter_scenes.py eval  --scene bounce|bh|orbit|globular --model E|bounce1|bounce4 --ckpt <pt> --tag zeroshot
  python scripts/scatter_scenes.py train --scene bounce|orbit|globular --model ... --init <pt|none> --budget 900 --tag ft
All coordinates here are CENTERED (sf.CENTER is added only when handing data to sf.to_tensors). Seed 4738 for training data, eval seeds 9100/9200/9300.
"""
import argparse
import json
import math
import os
import types

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

EVAL_SEEDS = (9100, 9200, 9300)
TRAIN_SEED = 4738
KS = (5, 10, 20, 50, 100)
EPS = 0.5
BOX = 15
BOX_R, BOX_K, BOX_G, BOX_DT, BOX_SUB, BOX_V = 0.75, 400.0, 9.0, 0.15, 8, 2.3
CKPT_DIR = "checkpoints_scenes"
OUT_DIR = "results/scenes"


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def nbody_truth(pos0, vel0, mass, steps, dev, dt=0.1, substeps=4, eps=EPS, c=None):
    """fp64 kick-drift-kick softened 2D gravity with masses (== gs.rollout_torch for unit masses). Returns numpy (T+1,N,2) pos, vel."""
    h = dt / substeps
    pos = torch.tensor(pos0, dtype=torch.float64, device=dev)
    vel = torch.tensor(vel0, dtype=torch.float64, device=dev)
    m = torch.tensor(mass, dtype=torch.float64, device=dev)
    n = pos.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=dev)

    def acc(p):
        d = p[None, :, :] - p[:, None, :]
        inv = (((d ** 2).sum(-1)) + eps ** 2) ** -1.5
        inv = inv.masked_fill(eye, 0.0) * m[None, :]
        return (d * inv[..., None]).sum(1)

    if c is not None:
        vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
        mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
        vf = lambda q: speed(q, c)
    else:
        mom = vel
        vf = lambda q: q
    ps, vs = [pos.cpu().numpy()], [vf(mom).cpu().numpy()]
    a = acc(pos)
    for _ in range(steps):
        for _ in range(substeps):
            mom = mom + 0.5 * h * a
            pos = pos + h * vf(mom)
            a = acc(pos)
            mom = mom + 0.5 * h * a
        ps.append(pos.cpu().numpy())
        vs.append(vf(mom).cpu().numpy())
    return np.stack(ps), np.stack(vs)


def _pen(x):
    return torch.where(x > 0.0, BOX_K * x * x, torch.zeros_like(x))


def bounce_forces(p, lo, hi):
    x, y = p[:, 0], p[:, 1]
    left_x = (x - lo) < BOX_R
    right_x = ~left_x & ((hi - x) < BOX_R)
    left_y = (y - lo) < BOX_R
    right_y = ~left_y & ((hi - y) < BOX_R)
    z = torch.zeros_like(x)
    fx = BOX_G + torch.where(left_x, _pen(BOX_R - (x - lo)), z) - torch.where(right_x, _pen(BOX_R - (hi - x)), z)
    fy = torch.where(left_y, _pen(BOX_R - (y - lo)), z) - torch.where(right_y, _pen(BOX_R - (hi - y)), z)
    d = p[None, :, :] - p[:, None, :]
    dist = d.norm(dim=-1)
    f = _pen(2 * BOX_R - dist)
    f.fill_diagonal_(0.0)
    f = f / dist.clamp(min=1e-9)
    return torch.stack([fx, fy], dim=1) + (f[..., None] * d).sum(0)


def bounce_truth(pos0, vel0, steps, dev, dt=BOX_DT, sub=BOX_SUB):
    """bounce.py physics (centered coords, box [-7,7]; symplectic Euler, penalty contact), fp64."""
    lo, hi = -(BOX - 1) / 2, (BOX - 1) / 2
    p = torch.tensor(pos0, dtype=torch.float64, device=dev)
    v = torch.tensor(vel0, dtype=torch.float64, device=dev)
    ps, vs = [p.cpu().numpy()], [v.cpu().numpy()]
    h = dt / sub
    for _ in range(steps):
        for _ in range(sub):
            v = v + bounce_forces(p, lo, hi) * h
            p = p + v * h
        ps.append(p.cpu().numpy())
        vs.append(v.cpu().numpy())
    return np.stack(ps), np.stack(vs)


def gen_bounce(rng, n_range=(2, 20)):
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    pos = rng.uniform(-(BOX - 1) / 2, (BOX - 1) / 2, (n, 2))
    vel = rng.uniform(-BOX_V, BOX_V, (n, 2))
    return dict(pos=pos, vel=vel, mass=np.ones(n))


def kepler_state(a, e, f, mu, rng):
    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(f))
    vr = np.sqrt(mu / p) * e * np.sin(f)
    vt = np.sqrt(mu / p) * (1 + e * np.cos(f))
    th = rng.uniform(0, 2 * np.pi)
    er = np.array([np.cos(th), np.sin(th)])
    et = np.array([-np.sin(th), np.cos(th)])
    return r * er, vr * er + vt * et


def gen_binary(rng):
    a, e = rng.uniform(3.0, 8.0), rng.uniform(0.0, 0.7)
    rel, vrel = kepler_state(a, e, rng.uniform(0, 2 * np.pi), 2.0, rng)
    off = rng.uniform(-3, 3, 2)
    pos = np.stack([off - 0.5 * rel, off + 0.5 * rel])
    vel = np.stack([-0.5 * vrel, 0.5 * vrel])
    return dict(pos=pos, vel=vel, mass=np.ones(2), kind="binary")


def gen_planetary(rng):
    M = rng.uniform(10.0, 30.0)
    n = int(rng.integers(3, 7))
    while True:
        a = np.sort(np.exp(rng.uniform(np.log(3.0), np.log(16.0), n)))
        if (a[1:] / a[:-1]).min() > 1.25:
            break
    m = np.exp(rng.uniform(np.log(0.01), np.log(0.5), n))
    pos, vel = [np.zeros(2)], [np.zeros(2)]
    for ai, mi in zip(a, m):
        rel, vrel = kepler_state(ai, rng.uniform(0.0, 0.4), rng.uniform(0, 2 * np.pi), M + mi, rng)
        if rng.random() < 0.5:
            er = rel / np.linalg.norm(rel)
            vrel = 2 * (vrel @ er) * er - vrel
        pos.append(rel)
        vel.append(vrel)
    pos, vel = np.array(pos), np.array(vel)
    mass = np.concatenate([[M], m])
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    pos -= (mass[:, None] * pos).sum(0) / mass.sum()
    return dict(pos=pos, vel=vel, mass=mass, kind="planetary")


def pair_energy(pos, vel, eps=EPS):
    d = pos[:, None] - pos[None]
    r = np.sqrt((d ** 2).sum(-1) + eps ** 2)
    iu = np.triu_indices(len(pos), 1)
    return 0.5 * (vel ** 2).sum(), -(1.0 / r[iu]).sum()


def gen_plummer(rng, n, a=4.0, rmax=28.0):
    pos = []
    while len(pos) < n:
        u = rng.uniform(0, 1)
        R = a * math.sqrt(u / (1 - u + 1e-12))
        if R < rmax:
            th = rng.uniform(0, 2 * np.pi)
            pos.append([R * math.cos(th), R * math.sin(th)])
    pos = np.array(pos)
    R = np.linalg.norm(pos, axis=1)
    sig = (1 + (R / a) ** 2) ** -0.25
    vel = rng.normal(size=(n, 2)) * sig[:, None]
    vel -= vel.mean(0)
    ke, pe = pair_energy(pos, vel)
    vel *= math.sqrt(-0.5 * pe / ke)
    return dict(pos=pos, vel=vel, mass=np.ones(n), kind="plummer")


def gen_bh(rng, n=300, sigma=8.0, vfac=0.3):
    pos = np.clip(rng.normal(0.0, sigma, (n, 2)), -29.0, 29.0)
    vel = rng.normal(0.0, vfac * np.sqrt(n / (4 * sigma)), (n, 2))
    vel -= vel.mean(0)
    return dict(pos=pos, vel=vel, mass=np.ones(n))


def pad_scenes(scenes, T, trajs):
    """trajs: list of (P,V) numpy (T+1,n,2) centered. Returns dict of tensors via sf.to_tensors (needs absolute coords)."""
    data = [(P + sf.CENTER, V, s["mass"]) for (P, V), s in zip(trajs, scenes)]
    return data


def model_rollout(model, scenes, steps, dev):
    n = max(len(s["mass"]) for s in scenes)
    S = len(scenes)
    P0 = torch.zeros(S, n, 2, device=dev)
    V0 = torch.zeros(S, n, 2, device=dev)
    M = torch.zeros(S, n, device=dev)
    mask = torch.zeros(S, n, device=dev)
    for i, s in enumerate(scenes):
        k = len(s["mass"])
        P0[i, :k] = torch.tensor(s["pos"], dtype=torch.float32)
        V0[i, :k] = torch.tensor(s["vel"], dtype=torch.float32)
        M[i, :k] = torch.tensor(s["mass"], dtype=torch.float32)
        mask[i, :k] = 1.0
    Pm, Vm = sf.rollout(model, P0, V0, M, mask, steps)
    return Pm.double().cpu().numpy(), Vm.double().cpu().numpy(), mask.cpu().numpy()


def model_rollout_bh(model, scene, steps, dt, c, dev):
    n = len(scene["mass"])
    pos = torch.tensor(scene["pos"], dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(scene["vel"], dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
    field = model.init_field(1, dev)[0]
    snaps, vm, vmax = [pos[0].cpu().numpy()], [speed(mom, c)[0].cpu().numpy()], []
    with torch.no_grad():
        a, field = model.force(pos, speed(mom, c), mass, mask, field)
        for _ in range(steps):
            mom = mom + 0.5 * dt * a
            pos = pos + dt * speed(mom, c)
            a, field = model.force(pos, speed(mom, c), mass, mask, field)
            mom = mom + 0.5 * dt * a
            snaps.append(pos[0].cpu().numpy())
            vm.append(speed(mom, c)[0].cpu().numpy())
            vmax.append(float(speed(mom, c).norm(dim=-1).max() / c))
    return np.stack(snaps).astype(np.float64), np.stack(vm).astype(np.float64), vmax


def build_model(name, dt, dev, ckpt=None):
    cfg = tsf.EXPS["E"]
    kw = dict(tsf.BASE)
    kw.update(tsf.VARIANTS["ms_kp_pot_v_g128"])
    if name == "E":
        model = sf.ScatterField(extent=cfg["extent"], dt=dt, in_scale=cfg["in_scale"], **kw)
    else:
        from scripts.scatter_bounce import BounceScatterField

        sub = int(name.replace("bounce", ""))
        model = BounceScatterField(extent=cfg["extent"], dt=dt, in_scale=cfg["in_scale"], sub=sub, **kw)
    if ckpt and ckpt != "none":
        sd = torch.load(ckpt, map_location=dev, weights_only=False)["model"]
        missing = model.load_state_dict(sd, strict=False)
        print(f"loaded {ckpt}: missing {list(missing.missing_keys)} unexpected {list(missing.unexpected_keys)}", flush=True)
    model.to(dev).eval()
    return model


def err_curve(Pm, Pt, mask):
    cnt = mask.sum(1)
    e = (np.linalg.norm(Pm - Pt, axis=-1) * mask[:, None]).sum(2) / cnt[:, None]
    return e.mean(0)


def at(curve, ks=KS):
    return {str(k): float(curve[k]) for k in ks if k < len(curve)}


def pad_traj(trajs, n):
    out = np.zeros((len(trajs), trajs[0].shape[0], n, 2))
    for i, t in enumerate(trajs):
        out[i, :, : t.shape[1]] = t
    return out


def perturb(scene, rel, rng):
    L = float(np.sqrt((scene["pos"] ** 2).sum(-1).mean()))
    return dict(scene, pos=scene["pos"] + rel * L * rng.normal(size=scene["pos"].shape))


def conservation(Pm, Vm, Pt, Vt, scenes, mask):
    n = Pm.shape[2]
    M = np.zeros((len(scenes), n))
    for i, s in enumerate(scenes):
        M[i, : len(s["mass"])] = s["mass"]
    t = lambda x: torch.tensor(x)
    return sf.traj_metrics(t(Pm), t(Vm), t(Pt), t(Vt), t(M), t(mask), EPS)


def finish(res, tag, scene, scenes_npz):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(f"{OUT_DIR}/{scene}_{tag}.json", "w"))
    np.savez_compressed(f"{OUT_DIR}/{scene}_{tag}_traj.npz", **scenes_npz)
    print(f"wrote {OUT_DIR}/{scene}_{tag}.json", flush=True)


def drift_at(m, key, k):
    x = m[key]
    return float(x[min(k, len(x) - 1)])


def eval_bounce(model, dev, tag, steps=100, n_scenes=48, floor=True):
    res = {"scene": "bounce", "steps": steps, "dt": BOX_DT, "seeds": {}}
    keep = {}
    for seed in EVAL_SEEDS:
        rng = np.random.default_rng(seed)
        scenes = [gen_bounce(rng) for _ in range(n_scenes)]
        trajs = [bounce_truth(s["pos"], s["vel"], steps, dev) for s in scenes]
        n = max(len(s["mass"]) for s in scenes)
        Pt, Vt = pad_traj([t[0] for t in trajs], n), pad_traj([t[1] for t in trajs], n)
        Pm, Vm, mask = model_rollout(model, scenes, steps, dev)
        t = np.arange(steps + 1)[None, :, None, None] * BOX_DT
        P0, V0 = Pt[:, :1], Vt[:, :1]
        cv = P0 + V0 * t
        ball = cv.copy()
        ball[..., 0] += 0.5 * BOX_G * t[..., 0] ** 2
        ballv = V0 + 0 * t
        r = {"err": at(err_curve(Pm, Pt, mask)), "const_vel": at(err_curve(cv, Pt, mask)), "ballistic_gravity": at(err_curve(ball, Pt, mask))}
        if floor:
            rngp = np.random.default_rng(seed + 1)
            pt = [bounce_truth(perturb(s, 1e-5, rngp)["pos"], s["vel"], steps, dev) for s in scenes]
            Pp = pad_traj([x[0] for x in pt], n)
            r["floor_pert1e-5"] = at(err_curve(Pp, Pt, mask))
        box = (BOX - 1) / 2 + 0.5
        out = (np.abs(Pm) > box).any(-1) * mask[:, None]
        r["outside_box_frac"] = {str(k): float((out[:, k].sum(1) / mask.sum(1)).mean()) for k in (20, 50, 100) if k <= steps}
        ke = lambda V: (0.5 * (V ** 2).sum(-1) * mask[:, None]).sum(2) / mask.sum(1)[:, None]
        r["ke_per_ball_model"] = [float(x) for x in ke(Vm).mean(0)[::10]]
        r["ke_per_ball_truth"] = [float(x) for x in ke(Vt).mean(0)[::10]]
        mx = lambda P: (P[..., 0] * mask[:, None]).sum(2) / mask.sum(1)[:, None]
        r["mean_x_model"] = [float(x) for x in mx(Pm).mean(0)[::10]]
        r["mean_x_truth"] = [float(x) for x in mx(Pt).mean(0)[::10]]
        Em = ke(Vm) - BOX_G * mx(Pm)
        Et = ke(Vt) - BOX_G * mx(Pt)
        r["dE_over_ke0_model"] = float((np.abs(Em[:, -1] - Em[:, 0]) / ke(Vt)[:, 0].clip(1e-6)).mean())
        r["dE_over_ke0_truth"] = float((np.abs(Et[:, -1] - Et[:, 0]) / ke(Vt)[:, 0].clip(1e-6)).mean())

        def overlap(P):
            d = np.linalg.norm(P[:, :, :, None] - P[:, :, None, :], axis=-1)
            mm = mask[:, None, :, None] * mask[:, None, None, :] * (1 - np.eye(n))[None, None]
            return float(((d < 2 * BOX_R * 0.7) * mm).sum() / max(mm.sum(), 1))

        r["deep_overlap_frac_model"], r["deep_overlap_frac_truth"] = overlap(Pm), overlap(Pt)
        res["seeds"][str(seed)] = r
        if seed == EVAL_SEEDS[0]:
            keep = dict(truth=Pt[:4], model=Pm[:4], mask=mask[:4], kind="bounce")
    res["restitution"] = restitution_probe(model, dev)
    res["mean"] = {k: {kk: float(np.mean([res["seeds"][s][k][kk] for s in res["seeds"]])) for kk in res["seeds"][str(EVAL_SEEDS[0])][k]}
                   for k in ("err", "const_vel", "ballistic_gravity", "outside_box_frac") + (("floor_pert1e-5",) if floor else ())}
    for k in ("dE_over_ke0_model", "dE_over_ke0_truth", "deep_overlap_frac_model", "deep_overlap_frac_truth"):
        res["mean"][k] = float(np.mean([res["seeds"][s][k] for s in res["seeds"]]))
    print("bounce err", res["mean"]["err"], "ballistic", res["mean"]["ballistic_gravity"], "outside", res["mean"]["outside_box_frac"], flush=True)
    print("restitution", res["restitution"], flush=True)
    return res, keep


def restitution_probe(model, dev, speeds=(2, 4, 6, 8, 10, 12, 15), steps=80):
    """Ball released at rest above the +x wall (gravity 9 toward +x); impact speed v = sqrt(2 g h). e_eff = sqrt(rebound height / drop height)."""
    out = {}
    wall = (BOX - 1) / 2 - BOX_R
    for v in speeds:
        x0 = wall - v * v / (2 * BOX_G)
        if x0 < -(BOX - 1) / 2:
            continue
        sc = [dict(pos=np.array([[x0, 0.0]]), vel=np.zeros((1, 2)), mass=np.ones(1))]
        Pt, _ = bounce_truth(sc[0]["pos"], sc[0]["vel"], steps, dev)
        Pm, _, _ = model_rollout(model, sc, steps, dev)
        res = {}
        for tag, P in (("truth", Pt[:, 0, 0]), ("model", Pm[0, :, 0, 0])):
            enter = np.nonzero(P > wall - 0.05)[0]
            if len(enter) == 0:
                res[tag] = None
                continue
            back = np.nonzero(P[enter[0]:] < wall - 0.05)[0]
            if len(back) == 0:
                res[tag] = None
                continue
            i_exit = enter[0] + back[0]
            apex = P[i_exit:].min()
            res[tag] = float(math.sqrt(max(wall - apex, 0.0) / max(wall - x0, 1e-9)))
        out[str(v)] = res
    return out


def orbit_stats(P, V, M):
    rel = P[:, 1:] - P[:, :1]
    vrel = V[:, 1:] - V[:, :1]
    th = np.unwrap(np.arctan2(rel[..., 1], rel[..., 0]), axis=0)
    mu = M[0] + M[1:]
    r = np.linalg.norm(rel, axis=-1)
    E = 0.5 * (vrel ** 2).sum(-1) - mu / np.sqrt(r ** 2 + EPS ** 2)
    h = rel[..., 0] * vrel[..., 1] - rel[..., 1] * vrel[..., 0]
    e = np.sqrt(np.clip(1 + 2 * E * h ** 2 / mu ** 2, 0, None))
    return th, r, E, h, e


def eval_orbit(model, dev, tag, steps=1000, n_scenes=24):
    res = {"scene": "orbit", "steps": steps, "seeds": {}}
    keep = {}
    ks = KS + (300, 1000)
    for fam, gen in (("binary", gen_binary), ("planetary", gen_planetary)):
        res[fam] = {"seeds": {}}
        for seed in EVAL_SEEDS:
            rng = np.random.default_rng(seed)
            scenes = [gen(rng) for _ in range(n_scenes)]
            trajs = [nbody_truth(s["pos"], s["vel"], s["mass"], steps, dev) for s in scenes]
            n = max(len(s["mass"]) for s in scenes)
            Pt, Vt = pad_traj([t[0] for t in trajs], n), pad_traj([t[1] for t in trajs], n)
            Pm, Vm, mask = model_rollout(model, scenes, steps, dev)
            rngp = np.random.default_rng(seed + 1)
            pt = [nbody_truth(perturb(s, 1e-5, rngp)["pos"], s["vel"], s["mass"], steps, dev) for s in scenes]
            Pp = pad_traj([x[0] for x in pt], n)
            r = {"err": at(err_curve(Pm, Pt, mask), ks), "floor_pert1e-5": at(err_curve(Pp, Pt, mask), ks)}
            cv = Pt[:, :1] + Vt[:, :1] * 0.1 * np.arange(steps + 1)[None, :, None, None]
            r["const_vel"] = at(err_curve(cv, Pt, mask), ks)
            cons = conservation(Pm, Vm, Pt, Vt, scenes, mask)
            r["dE_over_E_model_at"] = {str(k): float(cons["energy_drift_rel_model"][k]) for k in (100, 300, 1000)}
            r["dE_over_E_truth_at"] = {str(k): float(cons["energy_drift_rel_true"][k]) for k in (100, 300, 1000)}
            r["dL_over_L_model_at"] = {str(k): float(cons["angmom_drift_rel_model"][k]) for k in (100, 300, 1000)}
            r["dL_over_L_truth_at"] = {str(k): float(cons["angmom_drift_rel_true"][k]) for k in (100, 300, 1000)}
            per, lost, eorb_m, eorb_t, ecc_m, ecc_t, errt = [], [], [], [], [], [], []
            for i, s in enumerate(scenes):
                k = len(s["mass"])
                thm, rm, Em, hm, em = orbit_stats(Pm[i, :, :k], Vm[i, :, :k], s["mass"])
                tht, rt, Et, ht, et = orbit_stats(Pt[i, :, :k], Vt[i, :, :k], s["mass"])
                om_m, om_t = (thm[-1] - thm[0]) / steps, (tht[-1] - tht[0]) / steps
                per += list(np.abs(om_m / om_t - 1.0))
                gone = (rm.max(0) > 3 * rt.max(0)) | (np.abs(Pm[i, -1, :k, :][1:]).max(-1) > 32) | ~np.isfinite(rm[-1])
                lost += list(gone)
                eorb_m += list(np.abs(Em[-1] - Em[0]) / np.abs(Em[0]))
                eorb_t += list(np.abs(Et[-1] - Et[0]) / np.abs(Et[0]))
                ecc_m += list(np.abs(em[-1] - em[0]))
                ecc_t += list(np.abs(et[-1] - et[0]))
                errt += list(np.linalg.norm(Pm[i, 100, 1:k] - Pt[i, 100, 1:k], axis=-1) / np.maximum(rt[0], 1e-6))
            r["period_err_median"] = float(np.nanmedian(per))
            r["period_err_p90"] = float(np.nanpercentile(per, 90))
            r["planets_lost_frac"] = float(np.mean(lost))
            r["orbit_energy_drift_median_model"], r["orbit_energy_drift_median_truth"] = float(np.nanmedian(eorb_m)), float(np.nanmedian(eorb_t))
            r["ecc_drift_median_model"], r["ecc_drift_median_truth"] = float(np.nanmedian(ecc_m)), float(np.nanmedian(ecc_t))
            r["err100_over_r0_median"] = float(np.nanmedian(errt))
            res[fam]["seeds"][str(seed)] = r
            if seed == EVAL_SEEDS[0]:
                keep[fam + "_truth"], keep[fam + "_model"], keep[fam + "_mask"] = Pt[:4], Pm[:4], mask[:4]
        sd = res[fam]["seeds"]
        res[fam]["mean"] = {}
        for k, v in sd[str(EVAL_SEEDS[0])].items():
            if isinstance(v, dict):
                res[fam]["mean"][k] = {kk: float(np.mean([sd[s][k][kk] for s in sd])) for kk in v}
            else:
                res[fam]["mean"][k] = float(np.mean([sd[s][k] for s in sd]))
        print(fam, json.dumps(res[fam]["mean"]), flush=True)
    return res, keep


def cluster_stats(P, V, tick_step=10):
    """P,V numpy (T+1,N,2) unit masses. Lagrangian radii about the median centre, KE, PE, virial ratio, escapers, at every tick_step."""
    out = {"t": [], "r10": [], "r50": [], "r90": [], "ke": [], "pe": [], "e": [], "virial": [], "sigma": [], "esc": [], "outside_grid": []}
    for t in range(0, P.shape[0], tick_step):
        p, v = torch.tensor(P[t]), torch.tensor(V[t])
        c = p.median(0).values
        r = (p - c).norm(dim=-1)
        rs = r.sort().values
        n = len(rs)
        d = p[:, None] - p[None]
        rr = torch.sqrt((d ** 2).sum(-1) + EPS ** 2)
        inv = 1.0 / rr
        inv.fill_diagonal_(0.0)
        phi = -inv.sum(1)
        ke_i = 0.5 * (v ** 2).sum(-1)
        ke = float(ke_i.sum())
        pe = float(0.5 * phi.sum())
        r50 = float(rs[n // 2])
        esc = ((ke_i + phi > 0) & (r > 3 * r50)).float().mean()
        out["t"].append(t)
        out["r10"].append(float(rs[n // 10]))
        out["r50"].append(r50)
        out["r90"].append(float(rs[int(0.9 * n)]))
        out["ke"].append(ke)
        out["pe"].append(pe)
        out["e"].append(ke + pe)
        out["virial"].append(2 * ke / abs(pe))
        out["sigma"].append(float(((v - v.mean(0)) ** 2).sum(-1).mean().sqrt()))
        out["esc"].append(float(esc))
        out["outside_grid"].append(float((p.abs() > 32).any(-1).float().mean()))
    return out


def band_fraction(m, t, floor_rel, key):
    a, b, f = np.array(m[key]), np.array(t[key]), np.array(floor_rel[key])
    tol = np.maximum(2 * f, 0.15 * np.abs(b))
    return float((np.abs(a - b) <= tol).mean())


def eval_globular(model, dev, tag, steps=1000):
    res = {"scene": "globular", "steps": steps, "runs": {}}
    keep = {}
    for n, scenes_per_seed in ((300, 2), (100, 1), (1000, 1)):
        if n > 512:
            sf.apply_opt(model, "cell") if "cell" not in model.opt else None
        runs = []
        for seed in EVAL_SEEDS:
            rng = np.random.default_rng(seed)
            for j in range(scenes_per_seed):
                s = gen_plummer(rng, n)
                Pt, Vt = nbody_truth(s["pos"], s["vel"], s["mass"], steps, dev)
                pt2 = perturb(s, 1e-4, np.random.default_rng(seed + 1))
                Pp, Vp = nbody_truth(pt2["pos"], s["vel"], s["mass"], steps, dev)
                Pm, Vm, mask = model_rollout(model, [s], steps, dev)
                Pm, Vm = Pm[0], Vm[0]
                ct, cm, cp = cluster_stats(Pt, Vt), cluster_stats(Pm, Vm), cluster_stats(Pp, Vp)
                floor = {k: [abs(a - b) for a, b in zip(cp[k], ct[k])] for k in ("r50", "e")}
                run = {"seed": seed, "n": n, "err": at(np.linalg.norm(Pm - Pt, axis=-1).mean(1)), "err_floor_pert1e-4": at(np.linalg.norm(Pp - Pt, axis=-1).mean(1)),
                       "stats_model": cm, "stats_truth": ct, "stats_twin": cp,
                       "r50_band_frac": band_fraction(cm, ct, floor, "r50"), "e_band_frac": band_fraction(cm, ct, floor, "e"),
                       "dE_over_E_model_end": float(abs(cm["e"][-1] - cm["e"][0]) / abs(cm["e"][0])), "dE_over_E_truth_end": float(abs(ct["e"][-1] - ct["e"][0]) / abs(ct["e"][0])),
                       "esc_model_end": cm["esc"][-1], "esc_truth_end": ct["esc"][-1], "outside_grid_end": cm["outside_grid"][-1],
                       "r50_ratio_end": cm["r50"][-1] / ct["r50"][0], "r50_truth_ratio_end": ct["r50"][-1] / ct["r50"][0]}
                runs.append(run)
                if n == 300 and seed == EVAL_SEEDS[0] and j == 0:
                    keep = dict(truth=Pt[::10], model=Pm[::10], kind="globular")
                print(f"globular n={n} seed={seed} err {run['err']} r50_band {run['r50_band_frac']:.2f} e_band {run['e_band_frac']:.2f} dE/E {run['dE_over_E_model_end']:.3f} (truth {run['dE_over_E_truth_end']:.3f}) r50 end/0 model {run['r50_ratio_end']:.2f} truth {run['r50_truth_ratio_end']:.2f}", flush=True)
        res["runs"][str(n)] = runs
    return res, keep


def eval_bh(model, dev, tag, steps=100):
    res = {"scene": "bh", "steps": steps, "variants": {}}
    keep = {}
    variants = {"n300_c10": dict(n=300, sigma=8.0, c=10.0), "n100": dict(n=100, sigma=8.0, c=10.0), "n1000": dict(n=1000, sigma=8.0, c=10.0),
                "sigma5": dict(n=300, sigma=5.0, c=10.0), "sigma12": dict(n=300, sigma=12.0, c=10.0), "c5": dict(n=300, sigma=8.0, c=5.0), "c20": dict(n=300, sigma=8.0, c=20.0)}
    for name, v in variants.items():
        if v["n"] > 512 and "cell" not in model.opt:
            sf.apply_opt(model, "cell")
        runs = []
        for seed in EVAL_SEEDS + (TRAIN_SEED,):
            rng = np.random.default_rng(seed)
            s = gen_bh(rng, n=v["n"], sigma=v["sigma"])
            Pt, _ = nbody_truth(s["pos"], s["vel"], s["mass"], steps, dev, c=v["c"])
            Pm, vm, vmax = model_rollout_bh(model, s, steps, 0.1, v["c"], dev)
            err = np.linalg.norm(Pm - Pt, axis=-1).mean(1)
            pp = perturb(s, 1e-5, np.random.default_rng(seed + 1))
            Pp, _ = nbody_truth(pp["pos"], s["vel"], s["mass"], steps, dev, c=v["c"])
            fl = np.linalg.norm(Pp - Pt, axis=-1).mean(1)
            rm = [float(np.percentile(np.linalg.norm(f - np.median(f, 0), axis=1), 45)) for f in Pm]
            rt = [float(np.percentile(np.linalg.norm(f - np.median(f, 0), axis=1), 45)) for f in Pt]
            runs.append({"seed": seed, "err": at(err), "floor_pert1e-5": at(fl), "core_model": [rm[i] for i in (0, 10, 20, 50, 100)], "core_truth": [rt[i] for i in (0, 10, 20, 50, 100)],
                         "vmax_over_c": max(vmax), "outside_grid": float((np.abs(Pm[-1]) > 32).any(-1).mean())})
            if name == "n300_c10" and seed == EVAL_SEEDS[0]:
                keep = dict(truth=Pt, model=Pm, kind="bh")
        res["variants"][name] = {"runs": runs, "mean_err": {k: float(np.mean([r["err"][k] for r in runs])) for k in runs[0]["err"]},
                                 "mean_floor": {k: float(np.mean([r["floor_pert1e-5"][k] for r in runs])) for k in runs[0]["err"]}}
        print("bh", name, res["variants"][name]["mean_err"], "vmax/c", max(r["vmax_over_c"] for r in runs), flush=True)
    return res, keep


def make_train_data(scene, num, steps, seed, dev):
    rng = np.random.default_rng(seed)
    if scene == "bounce":
        scenes = [gen_bounce(rng) for _ in range(num)]
        trajs = [bounce_truth(s["pos"], s["vel"], steps, dev) for s in scenes]
        return pad_scenes(scenes, steps, trajs)
    if scene == "orbit":
        scenes = [gen_binary(rng) if i % 2 == 0 else gen_planetary(rng) for i in range(num)]
    else:
        scenes = [gen_plummer(rng, int(rng.integers(100, 301))) for _ in range(num)]
    trajs = [nbody_truth(s["pos"], s["vel"], s["mass"], steps, dev) for s in scenes]
    data = pad_scenes(scenes, steps, trajs)
    k = max(1, int(num * 0.25))
    cfg = dict(tsf.EXPS["E"])
    replay = tsf.make_data(cfg, k, steps, seed + 7, dev, 0.1, EPS)
    return data[k:] + replay


def run_train(args, dev):
    dt = BOX_DT if args.scene == "bounce" else 0.1
    model = build_model(args.model, dt, dev, args.init)
    model.train()
    for o in [x for x in args.opt.split(",") if x]:
        sf.apply_opt(model, o)
    if args.scene == "bounce" and args.init in (None, "none"):
        pass
    data = make_train_data(args.scene, args.train, args.steps, TRAIN_SEED, dev)
    tensors = sf.to_tensors(data, dev)
    print(f"train data {len(data)} scenes, N max {tensors[0].shape[2]}, params {sum(p.numel() for p in model.parameters())}", flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = f"{CKPT_DIR}/{args.scene}_{args.tag}.pt"
    targs = types.SimpleNamespace(lr=args.lr, device=str(dev), seed=TRAIN_SEED, opt=args.opt, time_budget=args.budget, iters=0, k_start=args.k_start,
                                  k_end=args.k_end, steps=args.steps, batch=args.batch, log_every=args.log_every)
    n_it, secs = tsf.train(model, tensors, targs, ckpt, lambda s: print(s, flush=True))
    print(f"TRAIN DONE iters {n_it} secs {secs:.0f}", flush=True)
    model.eval()
    return model, ckpt


EVALS = {"bounce": eval_bounce, "bh": eval_bh, "orbit": eval_orbit, "globular": eval_globular}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "train", "train_eval"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", default="E")
    ap.add_argument("--ckpt", default="checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--init", default="checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--opt", default="")
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-opt", default="kcache,fastio")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(TRAIN_SEED)
    scenes = args.scene.split(",")
    if args.mode == "eval":
        dt = BOX_DT if scenes[0] == "bounce" else 0.1
        model = build_model(args.model, dt, dev, args.ckpt)
    else:
        model, args.ckpt = run_train(args, dev)
    for o in [x for x in args.eval_opt.split(",") if x]:
        sf.apply_opt(model, o)
    if args.mode == "train":
        return
    for sc in scenes:
        res, keep = EVALS[sc](model, dev, args.tag)
        res.update(model=args.model, ckpt=args.ckpt, tag=args.tag)
        finish(res, args.tag, sc, keep)


if __name__ == "__main__":
    main()
