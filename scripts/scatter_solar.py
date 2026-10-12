"""Solar-system scene for the scatter-field models: star + 8 near-circular planets + a moon + a test-particle belt, softened 2D gravity (G=1, eps .5, dt .1), vs the exact fp64 sim on the same IC.

  python scripts/scatter_solar.py select --cands name=exp:variant:ckpt,...   # planetary benchmark (scatter_scenes.gen_planetary, seeds 9100/9200/9300, 24 scenes, 1000 ticks) per candidate checkpoint
  python scripts/scatter_solar.py run --cands ... --steps 2000                # solar IC, every candidate, metrics vs truth, trajectories to results/solar/
All coordinates are CENTERED like scatter_scenes.py. Seed 4738 for the solar IC.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import train_scatter_field as tsf

EPS = ss.EPS
OUT = "results/solar"
STAR_M = 20.0
RADII = (3.2, 4.5, 6.2, 8.2, 10.5, 13.0, 19.5, 23.0)
PMASS = (0.02, 0.05, 0.10, 0.05, 0.50, 0.30, 0.40, 0.08)
MOON_HOST, MOON_R, MOON_M = 5, 0.8, 0.01
BELT = (15.0, 17.5)
BELT_M = 0.001


def circ_speed(r, mu):
    return math.sqrt(mu * r * r / (r * r + EPS ** 2) ** 1.5)


def solar_scene(seed=4738, nbelt=200):
    rng = np.random.default_rng(seed)
    pos, vel, mass, kind = [np.zeros(2)], [np.zeros(2)], [STAR_M], [0]

    def unit():
        th = rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])

    for r, m in zip(RADII, PMASS):
        er, et = unit()
        pos.append(r * er)
        vel.append(circ_speed(r, STAR_M) * (1 + 0.02 * rng.normal()) * et)
        mass.append(m)
        kind.append(1)
    er, et = unit()
    pos.append(pos[MOON_HOST] + MOON_R * er)
    vel.append(vel[MOON_HOST] + circ_speed(MOON_R, PMASS[MOON_HOST - 1]) * et)
    mass.append(MOON_M)
    kind.append(2)
    for _ in range(nbelt):
        er, et = unit()
        r = rng.uniform(*BELT)
        pos.append(r * er)
        vel.append(circ_speed(r, STAR_M) * (1 + 0.01 * rng.normal()) * et)
        mass.append(BELT_M)
        kind.append(3)
    pos, vel, mass = np.array(pos), np.array(vel), np.array(mass)
    vel -= (mass[:, None] * vel).sum(0) / mass.sum()
    pos -= (mass[:, None] * pos).sum(0) / mass.sum()
    return dict(pos=pos, vel=vel, mass=mass, kind=np.array(kind))


def build(spec, dt, dev):
    exp, variant, ckpt = spec
    m = tsf.build(tsf.EXPS[exp], variant, dt).to(dev)
    sf.load_compat(m, torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    return m.eval()


def parse_cands(s):
    out = {}
    for item in s.split(","):
        name, rest = item.split("=")
        exp, variant, ckpt = rest.split(":")
        out[name] = (exp, variant, os.path.expandvars(ckpt))
    return out


def energy_L(P, V, mass):
    """P, V (T,N,2): total softened-gravity energy and angular momentum about the origin per tick."""
    T = P.shape[0]
    iu = np.triu_indices(P.shape[1], 1)
    mm = mass[iu[0]] * mass[iu[1]]
    ke = 0.5 * (mass[None] * (V ** 2).sum(-1)).sum(1)
    pe = np.zeros(T)
    for t in range(T):
        d = P[t][iu[0]] - P[t][iu[1]]
        pe[t] = -(mm / np.sqrt((d ** 2).sum(-1) + EPS ** 2)).sum()
    L = (mass[None] * (P[..., 0] * V[..., 1] - P[..., 1] * V[..., 0])).sum(1)
    return ke + pe, L


def select_one(model, truth, dev):
    res = {}
    for seed in ss.EVAL_SEEDS:
        scenes, Pt, Vt = truth[seed]
        Pm, Vm, mask = ss.model_rollout(model, scenes, Pt.shape[1] - 1, dev)
        lost, per, errt, eorb = [], [], [], []
        for i, s in enumerate(scenes):
            k = len(s["mass"])
            thm, rm, Em, hm, em = ss.orbit_stats(Pm[i, :, :k], Vm[i, :, :k], s["mass"])
            tht, rt, Et, ht, et = ss.orbit_stats(Pt[i, :, :k], Vt[i, :, :k], s["mass"])
            per += list(np.abs((thm[-1] - thm[0]) / (tht[-1] - tht[0]) - 1.0))
            gone = (rm.max(0) > 3 * rt.max(0)) | (np.abs(Pm[i, -1, 1:k]).max(-1) > 32) | ~np.isfinite(rm[-1])
            lost += list(gone)
            errt += list(np.linalg.norm(Pm[i, 100, 1:k] - Pt[i, 100, 1:k], axis=-1) / np.maximum(rt[0], 1e-6))
            eorb += list(np.abs(Em[-1] - Em[0]) / np.abs(Em[0]))
        cons = ss.conservation(Pm, Vm, Pt, Vt, scenes, mask)
        res[str(seed)] = {"period_err_median": float(np.nanmedian(per)), "planets_lost_frac": float(np.mean(lost)), "err100_over_r0_median": float(np.nanmedian(errt)),
                          "orbit_energy_drift_median": float(np.nanmedian(eorb)), "dE_over_E_1000": float(cons["energy_drift_rel_model"][-1]),
                          "dL_over_L_1000": float(cons["angmom_drift_rel_model"][-1])}
    res["mean"] = {k: float(np.mean([res[str(s)][k] for s in ss.EVAL_SEEDS])) for k in res[str(ss.EVAL_SEEDS[0])]}
    return res


def solar_metrics(Pm, Vm, Pt, Vt, scene, ticks):
    mass, kind = scene["mass"], scene["kind"]
    T = Pm.shape[0] - 1
    pl = np.where(kind == 1)[0]
    out = {"ticks": [t for t in ticks if t <= T]}

    def ang(P, idx):
        rel = P[:, idx] - P[:, :1]
        return np.unwrap(np.arctan2(rel[..., 1], rel[..., 0]), axis=0), np.linalg.norm(rel, axis=-1)

    for tag, idx in (("planets", pl), ("belt", np.where(kind == 3)[0])):
        thm, rm = ang(Pm, idx)
        tht, rt = ang(Pt, idx)
        out[tag] = {}
        for t in out["ticks"]:
            per = np.abs((thm[t] - thm[0]) / (tht[t] - tht[0]) - 1.0)
            lost = (rm[: t + 1].max(0) > 3 * rt[: t + 1].max(0)) | ~np.isfinite(rm[t]) | (np.abs(Pm[t, idx]).max(-1) > 32)
            out[tag][str(t)] = {"period_err_median": float(np.nanmedian(per)), "lost_frac": float(lost.mean()),
                                "pos_err_median": float(np.nanmedian(np.linalg.norm(Pm[t, idx] - Pt[t, idx], axis=-1))),
                                "radius_ratio_median": float(np.nanmedian(rm[t] / rt[t]))}
        if tag == "planets":
            out["per_planet"] = [{"a0": float(rt[0, j]), "mass": float(mass[i]), "period_err_T": float(np.abs((thm[-1] - thm[0]) / (tht[-1] - tht[0]) - 1.0)[j]),
                                  "r_over_r0_T_model": float(rm[-1, j] / rm[0, j]), "r_over_r0_T_truth": float(rt[-1, j] / rt[0, j]),
                                  "r_range_model": [float(rm[:, j].min()), float(rm[:, j].max())], "r_range_truth": [float(rt[:, j].min()), float(rt[:, j].max())],
                                  "lost": bool(rm.max(0)[j] > 3 * rt.max(0)[j] or not np.isfinite(rm[-1, j]))} for j, i in enumerate(pl)]
    mi = int(np.where(kind == 2)[0][0])
    dm = np.linalg.norm(Pm[:, mi] - Pm[:, MOON_HOST], axis=-1)
    dt_ = np.linalg.norm(Pt[:, mi] - Pt[:, MOON_HOST], axis=-1)
    out["moon"] = {"sep_model_max": float(np.nanmax(dm)), "sep_truth_max": float(dt_.max()), "bound_at_T_model": bool(dm[-1] < 2.0), "bound_at_T_truth": bool(dt_[-1] < 2.0)}
    Em, Lm = energy_L(Pm, Vm, mass)
    Et, Lt = energy_L(Pt, Vt, mass)
    out["conservation"] = {str(t): {"dE_model": float((Em[t] - Em[0]) / abs(Em[0])), "dE_truth": float((Et[t] - Et[0]) / abs(Et[0])),
                                    "dL_model": float((Lm[t] - Lm[0]) / abs(Lm[0])), "dL_truth": float((Lt[t] - Lt[0]) / abs(Lt[0]))} for t in out["ticks"]}
    out["star_drift_model"] = float(np.linalg.norm(Pm[-1, 0] - Pm[0, 0]))
    out["star_drift_truth"] = float(np.linalg.norm(Pt[-1, 0] - Pt[0, 0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "run"])
    ap.add_argument("--cands", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--belt", type=int, default=200)
    ap.add_argument("--tag", default="solar")
    ap.add_argument("--n-scenes", type=int, default=24)
    args = ap.parse_args()
    dev = torch.device("cuda")
    cands = parse_cands(args.cands)
    os.makedirs(OUT, exist_ok=True)
    if args.mode == "select":
        truth = {}
        for seed in ss.EVAL_SEEDS:
            rng = np.random.default_rng(seed)
            scenes = [ss.gen_planetary(rng) for _ in range(args.n_scenes)]
            trajs = [ss.nbody_truth(s["pos"], s["vel"], s["mass"], 1000, dev) for s in scenes]
            n = max(len(s["mass"]) for s in scenes)
            truth[seed] = (scenes, ss.pad_traj([t[0] for t in trajs], n), ss.pad_traj([t[1] for t in trajs], n))
        res = {}
        for name, spec in cands.items():
            res[name] = select_one(build(spec, 0.1, dev), truth, dev)
            print(name, json.dumps(res[name]["mean"]), flush=True)
            json.dump(res, open(f"{OUT}/{args.tag}_select.json", "w"), indent=1)
    else:
        scene = solar_scene(4738, args.belt)
        Pt, Vt = ss.nbody_truth(scene["pos"], scene["vel"], scene["mass"], args.steps, dev)
        ticks = [100, 300, 1000, 2000, 5000]
        res = {"n_bodies": len(scene["mass"]), "steps": args.steps, "kind_counts": {str(k): int((scene["kind"] == k).sum()) for k in range(4)}, "ic": {"star_mass": STAR_M, "radii": RADII, "masses": PMASS,
               "moon": [MOON_HOST, MOON_R, MOON_M], "belt": [BELT[0], BELT[1], args.belt, BELT_M], "seed": 4738}, "cands": {}}
        keep = {"truth": Pt.astype(np.float32), "truth_v": Vt.astype(np.float32), "mass": scene["mass"], "kind": scene["kind"]}
        for name, spec in cands.items():
            Pm, Vm, _ = ss.model_rollout(build(spec, 0.1, dev), [scene], args.steps, dev)
            Pm, Vm = Pm[0], Vm[0]
            res["cands"][name] = solar_metrics(Pm, Vm, Pt, Vt, scene, ticks)
            keep["model_" + name] = Pm.astype(np.float32)
            print(name, json.dumps({k: v for k, v in res["cands"][name]["planets"].items()}), flush=True)
            json.dump(res, open(f"{OUT}/{args.tag}.json", "w"), indent=1)
        np.savez_compressed(f"{OUT}/{args.tag}_traj.npz", **keep)


if __name__ == "__main__":
    main()
