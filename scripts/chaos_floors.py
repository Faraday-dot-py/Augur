"""Per-regime / per-experiment / per-scene-type reference-noise and chaos floors for every harness metric (decision #2).

Floors depend on the scene distribution only (no checkpoint). Per scene, the reference is the float64, base-substep exact sim; variants are float32 (centered, and absolute
coords for datasets generated around gs.CENTER), twice / half the substeps (sub_x2 = reference-noise floor, the largest term in chaos-floor-investigation.md), and initial
positions perturbed by rel * L * N(0,1) (L = rms radius, same direction for every rel) at 1e-6, 1e-5, 1e-4 (chaos floor).
Metrics at ticks ks: err (mean per-body position error vs reference), dE_rel / dP / dL_rel (each variant's own drift from its tick 0; the 'ref' row is the exact
sim's own drift), bound_gap (|bound fraction - reference|), virial_gap (|virial / reference virial - 1|). Scene means per group.
Groups: harness regimes two_body, expB, bh300 (same seeds/scenes as scripts/scatter_regress.py); experiments A, B, C, D, E_plain, E_cluster (tsf.EXPS configs, tsf.EVAL_SEEDS);
scene types binary, planetary, plummer, bounce (scatter_scenes.py generators and eval seeds) and the bh variants of scatter_scenes.eval_bh.

Usage: PYTHONPATH=. python scripts/chaos_floors.py --out results/chaos_floors.json [--groups two_body,expB,...]
"""
import argparse
import json
import subprocess
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import scatter_scenes as st
from scripts import train_scatter_field as tsf
from scripts.scatter_regress import BH, SCENES, SEEDS, bh_scene, state_summary

KS = (5, 10, 20, 50, 100)
KS_LONG = KS + (300, 1000)
RELS = (1e-6, 1e-5, 1e-4)
DT, EPS = 0.1, 0.5
DEV = torch.device("cuda")


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def simulate(sc, steps, dtype, sub, pos0=None, shift=0.0):
    """-> (P, V) numpy float64 (steps+1, N, 2); V is the coordinate velocity. Mass-aware kick-drift-kick (== gs.rollout_torch for unit masses, asserted in main),
    relativistic momentum state when sc['c'] is set, symplectic-Euler penalty-wall box physics (== st.bounce_truth) when sc['kind'] == 'bounce'."""
    p0 = sc["pos"] if pos0 is None else pos0
    pos = torch.tensor(p0 + shift, dtype=dtype, device=DEV)
    vel = torch.tensor(sc["vel"], dtype=dtype, device=DEV)
    ps, vs = [pos.double().cpu().numpy() - shift], [vel.double().cpu().numpy()]
    if sc["kind"] == "bounce":
        lo, hi = -(st.BOX - 1) / 2, (st.BOX - 1) / 2
        h = sc["dt"] / sub
        for _ in range(steps):
            for _ in range(sub):
                vel = vel + st.bounce_forces(pos, lo, hi) * h
                pos = pos + vel * h
            ps.append(pos.double().cpu().numpy() - shift)
            vs.append(vel.double().cpu().numpy())
        return np.stack(ps), np.stack(vs)
    c = sc["c"]
    m = torch.tensor(sc["mass"], dtype=dtype, device=DEV)
    n = pos.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=DEV)
    h = sc["dt"] / sub

    def acc(p):
        d = p[None, :, :] - p[:, None, :]
        inv = (((d ** 2).sum(-1)) + EPS ** 2) ** -1.5
        inv = inv.masked_fill(eye, 0.0) * m[None, :]
        return (d * inv[..., None]).sum(1)

    if c is not None:
        vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
        mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
        vf = lambda q: speed(q, c)
        vs = [vf(mom).double().cpu().numpy()]
    else:
        mom = vel
        vf = lambda q: q
    a = acc(pos)
    for _ in range(steps):
        for _ in range(sub):
            mom = mom + 0.5 * h * a
            pos = pos + h * vf(mom)
            a = acc(pos)
            mom = mom + 0.5 * h * a
        ps.append(pos.double().cpu().numpy() - shift)
        vs.append(vf(mom).double().cpu().numpy())
    return np.stack(ps), np.stack(vs)


def scenes_from_data(data, kind="nbody", c=None, dt=DT):
    out = []
    for d in data:
        out.append(dict(pos=d[0][0] - gs.CENTER, vel=d[1][0], mass=np.asarray(d[2]) if len(d) > 2 else np.ones(d[0].shape[1]), kind=kind, c=c, dt=dt, sub=4,
                        shift=gs.CENTER, steps=100, ks=KS))
    return out


def plain(scs, steps=100, ks=KS, shift=0.0, c=None, dt=DT, sub=4, kind="nbody"):
    return [dict(pos=s["pos"], vel=s["vel"], mass=s["mass"], kind=kind, c=c, dt=dt, sub=sub, shift=shift, steps=steps, ks=ks) for s in scs]


def group_scenes(name):
    cfgs = {"two_body": dict(ball_range=(2, 2), scale=False), "expB": dict(ball_range=(10, 100), scale=True)}
    if name in cfgs:
        out = []
        for seed in SEEDS:
            out += scenes_from_data(gs.make_dataset(SCENES, cfgs[name]["ball_range"], 1, seed, scale=cfgs[name]["scale"], device=DEV, dt=DT, eps=EPS))
        return out
    if name == "bh300":
        return plain([dict(zip(("pos", "vel"), bh_scene(seed)), mass=np.ones(BH["n"])) for seed in SEEDS], c=BH["c"])
    if name in ("A", "B", "C", "D"):
        out = []
        for seed in tsf.EVAL_SEEDS:
            out += scenes_from_data(tsf.make_data(tsf.EXPS[name], 48, 1, seed, DEV, DT, EPS))
        return out
    if name in ("E_plain", "E_cluster"):
        out = []
        for seed in tsf.EVAL_SEEDS:
            data = tsf.make_data(tsf.EXPS["E"], 48, 1, seed, DEV, DT, EPS)
            k = int(48 * tsf.EXPS["E"]["cluster"])
            out += scenes_from_data(data[: 48 - k] if name == "E_plain" else data[48 - k:])
        return out
    if name in ("binary", "planetary"):
        gen = st.gen_binary if name == "binary" else st.gen_planetary
        out = []
        for seed in st.EVAL_SEEDS:
            rng = np.random.default_rng(seed)
            out += [gen(rng) for _ in range(8)]
        return plain(out, steps=1000, ks=KS_LONG)
    if name == "plummer":
        out = []
        for n, per in ((300, 2), (100, 1), (1000, 1)):
            for seed in st.EVAL_SEEDS:
                rng = np.random.default_rng(seed)
                out += [st.gen_plummer(rng, n) for _ in range(per)]
        return plain(out, steps=1000, ks=KS_LONG)
    if name == "bounce":
        out = []
        for seed in st.EVAL_SEEDS:
            rng = np.random.default_rng(seed)
            out += [st.gen_bounce(rng) for _ in range(16)]
        return plain(out, kind="bounce", dt=st.BOX_DT, sub=st.BOX_SUB)
    if name.startswith("bh_"):
        v = BH_VARIANTS[name]
        out = []
        for seed in st.EVAL_SEEDS + (st.TRAIN_SEED,):
            out.append(st.gen_bh(np.random.default_rng(seed), n=v["n"], sigma=v["sigma"]))
        return plain(out, c=v["c"])
    raise ValueError(name)


BH_VARIANTS = {"bh_n100": dict(n=100, sigma=8.0, c=10.0), "bh_n1000": dict(n=1000, sigma=8.0, c=10.0), "bh_sigma5": dict(n=300, sigma=5.0, c=10.0),
               "bh_sigma12": dict(n=300, sigma=12.0, c=10.0), "bh_c5": dict(n=300, sigma=8.0, c=5.0), "bh_c20": dict(n=300, sigma=8.0, c=20.0)}
GROUPS = ["two_body", "expB", "bh300", "A", "B", "C", "D", "E_plain", "E_cluster", "binary", "planetary", "plummer", "bounce"] + list(BH_VARIANTS)


def traj_state(sc, P, V, ks):
    return {k: state_summary(P[k], V[k], sc["mass"], EPS, sc["c"]) for k in (0,) + tuple(ks)}


def variant_metrics(sc, P, V, Pref, ref_state, ks, own_state, walls):
    """P, V this variant; returns {metric: {k: value}}."""
    out = {"err": {str(k): float(np.linalg.norm(P[k] - Pref[k], axis=-1).mean()) for k in ks}}
    if walls:
        return out
    S = own_state
    out["dE_rel"] = {str(k): abs(S[k]["E"] - S[0]["E"]) / max(abs(S[0]["E"]), 1e-9) for k in ks}
    out["dP"] = {str(k): float(np.linalg.norm(S[k]["P"] - S[0]["P"])) for k in ks}
    out["dL_rel"] = {str(k): abs(S[k]["L"] - S[0]["L"]) / max(abs(S[0]["L"]), 1e-3) for k in ks}
    out["bound_gap"] = {str(k): abs(S[k]["bound_frac"] - ref_state[k]["bound_frac"]) for k in ks}
    out["virial_gap"] = {str(k): abs(S[k]["virial"] / ref_state[k]["virial"] - 1) for k in ks}
    return out


def run_scene(sc):
    steps, ks, sub = sc["steps"], sc["ks"], sc["sub"]
    walls = sc["kind"] == "bounce"
    L = float(np.sqrt((sc["pos"] ** 2).sum(-1).mean()))
    direction = np.random.default_rng(st.TRAIN_SEED).normal(size=sc["pos"].shape)
    Pref, Vref = simulate(sc, steps, torch.float64, sub)
    ref_state = None if walls else traj_state(sc, Pref, Vref, ks)
    res = {"ref": variant_metrics(sc, Pref, Vref, Pref, ref_state, ks, ref_state, walls)}
    variants = [("f32", dict(dtype=torch.float32)), ("sub_x2", dict(dtype=torch.float64, sub=sub * 2)), ("sub_d2", dict(dtype=torch.float64, sub=max(sub // 2, 1)))]
    if sc["shift"]:
        variants.insert(1, ("f32abs", dict(dtype=torch.float32, shift=sc["shift"])))
    variants += [(f"pert_{rel:g}", dict(dtype=torch.float64, pos0=sc["pos"] + rel * L * direction)) for rel in RELS]
    for name, kw in variants:
        kw = dict(kw)
        dtype = kw.pop("dtype")
        P, V = simulate(sc, steps, dtype, kw.pop("sub", sub), **kw)
        res[name] = variant_metrics(sc, P, V, Pref, ref_state, ks, None if walls else traj_state(sc, P, V, ks), walls)
    return res


def check_reimplementation():
    sc = group_scenes("two_body")[0]
    sc.update(steps=20)
    P, _ = simulate(sc, 20, torch.float64, 4)
    ref = gs.rollout_torch(sc["pos"] + gs.CENTER, sc["vel"], 20, DEV, dt=DT, substeps=4, eps=EPS)[0] - gs.CENTER
    out = {"two_body_vs_gs": float(np.abs(P - ref).max())}
    sc = group_scenes("bh300")[0]
    P, _ = simulate(sc, 20, torch.float64, 4)
    ref = gs.rollout_torch(sc["pos"], sc["vel"], 20, DEV, dt=DT, substeps=4, eps=EPS, relativistic=True, c=BH["c"])[0]
    out["bh300_vs_gs"] = float(np.abs(P - ref).max())
    sc = group_scenes("C")[0]
    P, _ = simulate(sc, 20, torch.float64, 4)
    ref = sf.rollout_mass_torch(sc["pos"] + gs.CENTER, sc["vel"], sc["mass"], 20, DEV, dt=DT, eps=EPS)[0] - gs.CENTER
    out["star_vs_sf"] = float(np.abs(P - ref).max())
    sc = group_scenes("bounce")[0]
    P, _ = simulate(sc, 20, torch.float64, sc["sub"])
    ref = st.bounce_truth(sc["pos"], sc["vel"], 20, DEV)[0]
    out["bounce_vs_st"] = float(np.abs(P - ref).max())
    for k, v in out.items():
        print(f"reimplementation {k}: max |diff| {v:.3g}", flush=True)
        assert v < 1e-7, k
    return out


def aggregate(recs):
    out = {}
    for var in recs[0]:
        out[var] = {}
        for metric in recs[0][var]:
            out[var][metric] = {k: float(np.mean([r[var][metric][k] for r in recs])) for k in recs[0][var][metric]}
    out["_sem_err"] = {}
    for var in recs[0]:
        out["_sem_err"][var] = {k: float(np.std([r[var]["err"][k] for r in recs]) / np.sqrt(len(recs))) for k in recs[0][var]["err"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default=",".join(GROUPS))
    args = ap.parse_args()
    torch.manual_seed(4738)
    res = {"meta": {"git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "gpu": torch.cuda.get_device_name(DEV), "rels": RELS, "dt": DT, "eps": EPS, "reference": "float64, base substeps (4; bounce 8)",
                    "reference_noise_floor": "sub_x2", "chaos_floor": "pert_1e-05"},
           "reimplementation": check_reimplementation(), "groups": {}}
    for name in args.groups.split(","):
        t0 = time.time()
        scenes = group_scenes(name)
        recs = []
        for i, sc in enumerate(scenes):
            recs.append(run_scene(sc))
            if i % 10 == 0:
                print(f"{name} scene {i}/{len(scenes)} N {len(sc['mass'])} sub_x2 err@{sc['ks'][3]} {recs[-1]['sub_x2']['err'][str(sc['ks'][3])]:.3g}", flush=True)
        res["groups"][name] = {"n_scenes": len(scenes), "n_bodies": [int(min(len(s["mass"]) for s in scenes)), int(max(len(s["mass"]) for s in scenes))],
                               "steps": scenes[0]["steps"], "ks": list(scenes[0]["ks"]), "variants": aggregate(recs)}
        v = res["groups"][name]["variants"]
        print(f"{name}: {len(scenes)} scenes, {time.time() - t0:.0f} s; err@{list(scenes[0]['ks'])}", flush=True)
        for var in ("f32", "sub_x2", "pert_1e-05"):
            print(f"  {var:>11}", [float(f"{x:.3g}") for x in v[var]["err"].values()], flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
