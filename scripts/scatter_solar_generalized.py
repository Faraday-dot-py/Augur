"""Generalized planetary-system scenes for the scatter-field models (opt-in; scatter_solar.solar_scene is untouched).

Families (splits fixed before training, see docs/debugging/experiment-log.md):
  train-mixture       3-8 planets, Hill spacing k 4-8, planet/star mass ratio 3e-5..1.2e-3 (log-uniform), star mass 10-60, speed jitter 0.5-5%, 15% of scenes with retrograde planets,
                      60% one moon around an inner planet, 70% a belt (40-300 particles, mass 1e-4..2e-3, variable width and position)
  heavy-mass-ratio    held out: planet/star mass ratio 1.2e-3..3e-3, 3-6 planets
  many-planets        held out: 9-10 planets (k 4-5)
  circumbinary-star   held out: planets orbit a close equal-ish binary (q 0.3-1, separation 1.2-2)
  two-moons           held out: one planet (any, outer included) with two moons
  train-more-moon-situations   training only, never evaluated: train-mixture with 25% moon-free scenes, else 1-3 moon hosts (any planet) with 1-3 moons each, moon/host mass 3e-4..8e-2, orbit 0.06-0.45 Hill radius
Bodies are ordered star(s), planets, moons, belt. kind: 0 star, 4 secondary star, 1 planet, 2 moon, 3 belt. Exact fp64 integrator rejects unstable ICs.
"""
import math

import numpy as np

from scripts import scatter_scenes as ss
from scripts import scatter_solar as sol

EPS = ss.EPS
TRAIN_SEED_BASE = 100000
MAX_OUTER_R = 28.0
MIN_MOON_HILL = 0.25
MIN_TWO_MOON_HILL = 0.6
TRAIN_FAMILY = "train-mixture"
MORE_MOONS_FAMILY = "train-more-moon-situations"
MOON_HILL_MIN = {1: 0.25, 2: 0.6, 3: 0.9}
HELD_OUT_FAMILIES = ("heavy-mass-ratio", "many-planets", "circumbinary-star", "two-moons")
EVAL_FAMILIES = (TRAIN_FAMILY,) + HELD_OUT_FAMILIES
BASE = dict(star=(10.0, 60.0), n=(3, 8), ratio=(3e-5, 1.2e-3), k=(4.0, 8.0), r0=(2.0, 3.5), ecc=(0.005, 0.05), retro_p=0.15, belt_p=0.7, belt_n=(40, 300), belt_m=(1e-4, 2e-3),
            belt_w=(0.3, 0.8), moon_p=0.6, moon_n=1, moon_host="inner", moon_frac=(0.1, 0.4), moon_mratio=(1e-3, 5e-2), binary=False)
FAMILY_SPEC = {
    TRAIN_FAMILY: BASE,
    "heavy-mass-ratio": dict(BASE, ratio=(1.2e-3, 3e-3), n=(3, 6), k=(3.5, 6.0)),
    "many-planets": dict(BASE, n=(9, 10), k=(4.0, 5.0)),
    "circumbinary-star": dict(BASE, binary=True, n=(3, 7), k=(4.0, 7.0), moon_p=0.3),
    "two-moons": dict(BASE, moon_p=1.0, moon_n=2, moon_host="any"),
    MORE_MOONS_FAMILY: dict(BASE, moon_plan_p=0.75, moon_hosts=(0.55, 0.30, 0.15), moon_counts=(0.5, 0.35, 0.15), moon_frac=(0.06, 0.45), moon_mratio=(3e-4, 8e-2)),
}


def circ(r, mu):
    return sol.circ_speed(r, mu)


def hill(r, m, M):
    return r * (m / (3 * M)) ** (1 / 3)


def sample_ic(rng, sp):
    """One random system; returns scene dict (pos, vel, mass, kind, host, hill) or None if the Hill chain does not fit."""
    star_m = rng.uniform(*sp["star"])
    n = int(rng.integers(sp["n"][0], sp["n"][1] + 1))
    pm = np.exp(rng.uniform(np.log(sp["ratio"][0]), np.log(sp["ratio"][1]), n)) * star_m
    moon_hi = None
    plan = sp.get("moon_plan")
    moon_his = []
    if plan:
        moon_his = [int(i) for i in rng.choice(n, size=min(len(plan), n), replace=False)]
        for hi in moon_his:
            pm[hi] = star_m * math.exp(rng.uniform(math.log(0.4 * sp["ratio"][1]), math.log(sp["ratio"][1])))
    elif sp["moon_p"] >= 1.0:
        moon_hi = int(rng.integers(0, n if sp["moon_host"] == "any" else max(1, n - 2)))
        pm[moon_hi] = star_m * math.exp(rng.uniform(math.log(0.4 * sp["ratio"][1]), math.log(sp["ratio"][1])))
    k = rng.uniform(*sp["k"])
    ks = k * (1 + 0.15 * rng.uniform(-1, 1, n - 1))
    unit = lambda: (lambda th: (np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])))(rng.uniform(0, 2 * math.pi))
    binary = sp["binary"]
    if binary:
        q = rng.uniform(0.3, 1.0)
        ab = rng.uniform(1.2, 2.0)
        m1 = star_m / (1 + q)
        m2 = q * m1
        er, et = unit()
        vrel = circ(ab, star_m)
        stars = [(-m2 / star_m * ab * er, m2 / star_m * vrel * et, m1, 0), (m1 / star_m * ab * er, -m1 / star_m * vrel * et, m2, 4)]
        R0 = ab * rng.uniform(3.5, 4.5)
    else:
        stars = [(np.zeros(2), np.zeros(2), star_m, 0)]
        R0 = rng.uniform(*sp["r0"])
    has_belt = n >= 2 and sp["belt_p"] >= 1.0
    gap_after = int(rng.integers(0, n - 1)) if has_belt else -1
    bw = rng.uniform(*sp["belt_w"])
    r, belt = [R0], None
    for i in range(n - 1):
        x = ks[i] * ((pm[i] + pm[i + 1]) / (3 * star_m)) ** (1 / 3) / 2
        if x >= 0.9:
            return None
        rho = (1 + x) / (1 - x)
        if i == gap_after:
            gs = 0.55 + bw + 0.25
            belt = (r[-1] * rho ** 0.55, r[-1] * rho ** (0.55 + bw))
            r.append(r[-1] * rho ** gs)
        else:
            r.append(r[-1] * rho)
    if r[-1] > MAX_OUTER_R:
        return None
    sig = rng.uniform(*sp["ecc"])
    retro_scene = rng.random() < sp["retro_p"]
    pos, vel, mass, kind, host, hl = [], [], [], [], [], []
    for p, v, m, kd in stars:
        pos.append(p); vel.append(v); mass.append(m); kind.append(kd); host.append(-1); hl.append(0.0)
    pidx = []
    for ri, mi in zip(r, pm):
        er, et = unit()
        sgn = -1.0 if (retro_scene and rng.random() < 0.5) else 1.0
        pos.append(ri * er)
        vel.append(sgn * circ(ri, star_m) * (1 + sig * rng.normal()) * et)
        mass.append(mi); kind.append(1); host.append(-1); hl.append(hill(ri, mi, star_m))
        pidx.append(len(pos) - 1)
    for hi, nm in zip(moon_his, plan or []):
        hh = hill(r[hi], pm[hi], star_m)
        if hh < MOON_HILL_MIN[nm]:
            return None
        h = pidx[hi]
        fr = [rng.uniform(sp["moon_frac"][0], min(sp["moon_frac"][1], 0.15))] if nm > 1 else [rng.uniform(*sp["moon_frac"])]
        for _ in range(nm - 1):
            fr.append(fr[-1] * rng.uniform(1.6, 2.2))
        if fr[-1] > sp["moon_frac"][1]:
            return None
        for f in fr:
            Rm = max(0.1, f * hh)
            er, et = unit()
            mm = pm[hi] * math.exp(rng.uniform(math.log(sp["moon_mratio"][0]), math.log(sp["moon_mratio"][1])))
            sgn = -1.0 if rng.random() < 0.2 else 1.0
            pos.append(pos[h] + Rm * er)
            vel.append(vel[h] + sgn * circ(Rm, pm[hi]) * et)
            mass.append(mm); kind.append(2); host.append(h); hl.append(hh)
    if moon_hi is not None:
        hi = moon_hi
        if hill(r[hi], pm[hi], star_m) < (MIN_MOON_HILL if sp["moon_n"] == 1 else MIN_TWO_MOON_HILL):
            return None
        h = pidx[hi]
        hh = hill(r[hi], pm[hi], star_m)
        fr = sorted(rng.uniform(*sp["moon_frac"], sp["moon_n"])) if sp["moon_n"] == 1 else [rng.uniform(0.08, 0.18), rng.uniform(0.28, 0.4)]
        for f in fr:
            Rm = max(0.1, f * hh)
            er, et = unit()
            mm = pm[hi] * math.exp(rng.uniform(math.log(sp["moon_mratio"][0]), math.log(sp["moon_mratio"][1])))
            sgn = -1.0 if rng.random() < 0.2 else 1.0
            pos.append(pos[h] + Rm * er)
            vel.append(vel[h] + sgn * circ(Rm, pm[hi]) * et)
            mass.append(mm); kind.append(2); host.append(h); hl.append(hh)
    if has_belt:
        nb = int(rng.integers(sp["belt_n"][0], sp["belt_n"][1] + 1))
        bm = math.exp(rng.uniform(math.log(sp["belt_m"][0]), math.log(sp["belt_m"][1])))
        for _ in range(nb):
            er, et = unit()
            rb = rng.uniform(*belt)
            pos.append(rb * er)
            vel.append(circ(rb, star_m) * (1 + 0.01 * rng.normal()) * et)
            mass.append(bm); kind.append(3); host.append(-1); hl.append(0.0)
    pos, vel, mass = np.array(pos), np.array(vel), np.array(mass)
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    pos -= (mass[:, None] * pos).sum(0) / mass.sum()
    return dict(pos=pos, vel=vel, mass=mass, kind=np.array(kind), host=np.array(host), hill=np.array(hl), star_mass=star_m)


def star_center(P, scene):
    st = np.isin(scene["kind"], (0, 4))
    w = scene["mass"][st]
    return (P[:, st] * w[None, :, None]).sum(1) / w.sum()


def stability(P, scene, ticks):
    """None if the exact trajectory over the first `ticks` ticks is stable, else the rejection reason."""
    P = P[: ticks + 1]
    if not np.isfinite(P).all():
        return "nonfinite"
    kind = scene["kind"]
    rel = np.linalg.norm(P - star_center(P, scene)[:, None], axis=-1)
    pl = kind == 1
    if (rel[:, pl].max(0) > 2.0 * rel[0, pl]).any() or (rel[:, pl].min(0) < 0.5 * rel[0, pl]).any():
        return "planet"
    for j in np.where(kind == 2)[0]:
        if np.linalg.norm(P[:, j] - P[:, scene["host"][j]], axis=-1).max() > scene["hill"][j]:
            return "moon"
    bl = kind == 3
    if bl.any():
        bad = (rel[:, bl].max(0) > 2.0 * rel[0, bl]) | (rel[:, bl].min(0) < 0.5 * rel[0, bl])
        if bad.mean() > 0.3:
            return "belt"
    return None


def resolve_spec(rng, family):
    """Moon and belt presence are drawn once per scene (so rejections do not bias the accepted fraction); sample_ic then treats moon_p / belt_p as 1 or 0."""
    sp = dict(FAMILY_SPEC[family])
    if "moon_plan_p" in sp:
        sp["moon_p"] = 0.0
        if rng.random() < sp["moon_plan_p"]:
            nh = 1 + int(rng.choice(len(sp["moon_hosts"]), p=sp["moon_hosts"]))
            sp["moon_plan"] = [1 + int(rng.choice(len(sp["moon_counts"]), p=sp["moon_counts"])) for _ in range(nh)]
        sp["belt_p"] = 1.0 if rng.random() < sp["belt_p"] else 0.0
        return sp
    sp["moon_p"] = 1.0 if rng.random() < sp["moon_p"] else 0.0
    sp["belt_p"] = 1.0 if rng.random() < sp["belt_p"] else 0.0
    return sp


def make_scene(rng, family, steps, stab_ticks, dev, stats=None):
    """Sample until the exact integrator keeps the IC stable over stab_ticks; returns scene, truth P, V (steps+1, N, 2). stats: dict of counters (attempts, accepted, rejected_<reason>)."""
    sp = resolve_spec(rng, family)
    while True:
        st = stats if stats is not None else {}
        st["attempts"] = st.get("attempts", 0) + 1
        sc = sample_ic(rng, sp)
        if sc is None:
            st["rejected_radius"] = st.get("rejected_radius", 0) + 1
            continue
        T = max(stab_ticks, 1)
        Pt, Vt = ss.nbody_truth(sc["pos"], sc["vel"], sc["mass"], T, dev)
        why = stability(Pt, sc, stab_ticks)
        if why:
            st["rejected_" + why] = st.get("rejected_" + why, 0) + 1
            continue
        if steps > T:
            Pt, Vt = ss.nbody_truth(sc["pos"], sc["vel"], sc["mass"], steps, dev)
        st["accepted"] = st.get("accepted", 0) + 1
        return sc, Pt, Vt


def train_rng(i):
    return np.random.default_rng(TRAIN_SEED_BASE + i)


def eval_rng(seed, family, j):
    return np.random.default_rng([seed, EVAL_FAMILIES.index(family) + 1, j])


def energy_L_at(P, V, mass, ticks):
    """Softened-gravity energy and angular momentum about the origin at the listed ticks (sol.energy_L on that subset)."""
    return sol.energy_L(P[ticks], V[ticks], mass)


def family_metrics(Pm, Vm, Pt, Vt, scene, ticks):
    """Per-scene metrics vs the exact trajectory. Center = mass-weighted star position (the single star for one-star systems)."""
    mass, kind, host, hl = scene["mass"], scene["kind"], scene["host"], scene["hill"]
    T = Pm.shape[0] - 1
    ticks = [t for t in ticks if t <= T]
    cm, ct = star_center(Pm, scene), star_center(Pt, scene)
    pl, bl, mo = np.where(kind == 1)[0], np.where(kind == 3)[0], np.where(kind == 2)[0]

    def ang(P, c, idx):
        rel = P[:, idx] - c[:, None]
        return np.unwrap(np.arctan2(rel[..., 1], rel[..., 0]), axis=0), np.linalg.norm(rel, axis=-1)

    out = {"ticks": ticks, "n_planets": int(len(pl)), "n_bodies": int(len(mass))}
    for tag, idx in (("planets", pl), ("belt", bl)):
        if len(idx) == 0:
            continue
        thm, rm = ang(Pm, cm, idx)
        tht, rt = ang(Pt, ct, idx)
        out[tag] = {}
        for t in ticks:
            per = np.abs((thm[t] - thm[0]) / (tht[t] - tht[0]) - 1.0)
            lost = (rm[: t + 1].max(0) > 3 * rt[: t + 1].max(0)) | ~np.isfinite(rm[t]) | (np.abs(Pm[t, idx]).max(-1) > 32)
            tlost = rt[: t + 1].max(0) > 3 * rt[0]
            out[tag][str(t)] = {"period_err_median": float(np.nanmedian(per)), "lost_frac": float(lost.mean()), "kept": float(len(idx) * (1 - lost.mean())), "truth_lost_frac": float(tlost.mean())}
    out["channels"] = {}
    for tag, idx in (("star", np.where(np.isin(kind, (0, 4)))[0]), ("planets", pl), ("moons", mo), ("belt", bl)):
        if len(idx) == 0:
            continue
        out["channels"][tag] = {str(t): {"pos_err_median": float(np.nanmedian(np.linalg.norm(Pm[t, idx] - Pt[t, idx], axis=-1))),
                                         "vel_err_median": float(np.nanmedian(np.linalg.norm(Vm[t, idx] - Vt[t, idx], axis=-1)))} for t in ticks}
    out["moons"] = [{"bound_model_at_T": bool(np.linalg.norm(Pm[-1, j] - Pm[-1, host[j]]) < hl[j]), "bound_truth_at_T": bool(np.linalg.norm(Pt[-1, j] - Pt[-1, host[j]]) < hl[j])} for j in mo]
    tt = [0] + ticks
    Em, Lm = energy_L_at(Pm, Vm, mass, tt)
    Et, Lt = energy_L_at(Pt, Vt, mass, tt)
    out["conservation"] = {str(t): {"dE_model": float((Em[i + 1] - Em[0]) / abs(Em[0])), "dE_truth": float((Et[i + 1] - Et[0]) / abs(Et[0])),
                                    "dL_model": float((Lm[i + 1] - Lm[0]) / abs(Lm[0])), "dL_truth": float((Lt[i + 1] - Lt[0]) / abs(Lt[0]))} for i, t in enumerate(ticks)}
    return out
