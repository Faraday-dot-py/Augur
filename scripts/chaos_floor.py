"""Chaos-floor analysis for the scatter-field model: how much of err@k is irreducible divergence of the exact sim?

Exact reference = scripts/gravity_sim.rollout_torch math (kick-drift-kick, 4 substeps per tick, softened 2D gravity eps 0.5, dt 0.1; BH: relativistic momentum
state, c=10). Here re-implemented with selectable dtype/substeps and per-substep minimum pair distance; the float64/4-substep case is asserted equal to gs.rollout_torch.
Variants per scene: ref (f64, 4 sub), pert_<rel> (initial positions + rel * L * gaussian, L = rms radius of the scene; same direction for all magnitudes),
f32 (centered float32), f32abs (float32 in the dataset's absolute coords, centre 500; expb only), sub8 / sub2 (halved / doubled step).
All compared against ref with err@k = mean over bodies of |dx|, averaged over scenes. Model error on the same scenes from the checkpoint.

Usage: PYTHONPATH=. python scripts/chaos_floor.py --mode bh --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --tag chaos_bh
       PYTHONPATH=. python scripts/chaos_floor.py --mode expb --ckpt <expB ckpt> --tag chaos_expb
"""
import argparse
import json

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

RELS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
KS = (5, 10, 20, 50, 100)
RC = 0.5


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def sim(pos0, vel0, steps, dtype, substeps=4, dt=0.1, eps=0.5, c=None, shift=None):
    """pos0, vel0 (N,2) float64 numpy -> (pos (T+1,N,2) float64 numpy, mind (T,) min pair distance seen at substeps per tick, per body (T,N))."""
    dev = torch.device("cuda")
    h = dt / substeps
    pos = torch.tensor(pos0, dtype=dtype, device=dev)
    vel = torch.tensor(vel0, dtype=dtype, device=dev)
    if shift is not None:
        pos = pos - shift
    n = pos.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=dev)

    def acc(p):
        d = p[None, :, :] - p[:, None, :]
        r2 = (d ** 2).sum(-1)
        inv = (r2 + eps ** 2) ** -1.5
        inv = inv.masked_fill(eye, 0.0)
        return (d * inv[..., None]).sum(1), r2.masked_fill(eye, float("inf")).min(1).values

    if c is not None:
        vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
        mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
        vf = lambda m: speed(m, c)
    else:
        mom = vel
        vf = lambda m: m
    ps = [pos.double().cpu().numpy() + (0 if shift is None else float(shift))]
    md = []
    a, _ = acc(pos)
    for _ in range(steps):
        m2 = torch.full((n,), float("inf"), dtype=dtype, device=dev)
        for _ in range(substeps):
            mom = mom + 0.5 * h * a
            pos = pos + h * vf(mom)
            a, r2 = acc(pos)
            m2 = torch.minimum(m2, r2)
            mom = mom + 0.5 * h * a
        ps.append(pos.double().cpu().numpy() + (0 if shift is None else float(shift)))
        md.append(m2.double().sqrt().cpu().numpy())
    return np.stack(ps), np.stack(md)


def body_err(a, b):
    return np.linalg.norm(a - b, axis=-1)


def load_model(exp, ckpt, dt, dev):
    model = tsf.build(tsf.EXPS[exp], "ms_kp_pot_v_g128", dt).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    return model


def model_rollout_bh(model, pos0, vel0, steps, dt, c, dev):
    n = len(pos0)
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
    field = model.init_field(1, dev)[0]
    snaps = [pos[0].cpu().numpy()]
    with torch.no_grad():
        a, field = model.force(pos, speed(mom, c), mass, mask, field)
        for _ in range(steps):
            mom = mom + 0.5 * dt * a
            pos = pos + dt * speed(mom, c)
            a, field = model.force(pos, speed(mom, c), mass, mask, field)
            mom = mom + 0.5 * dt * a
            snaps.append(pos[0].cpu().numpy())
    return np.stack(snaps).astype(np.float64)


def model_rollout_expb(model, pos0c, vel0, steps, dev):
    pos = torch.tensor(pos0c, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    n = pos.shape[1]
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    with torch.no_grad():
        Pm, _ = sf.rollout(model, pos, vel, mass, mask, steps)
    return Pm[0].double().cpu().numpy()


def scene_record(name, pos0, vel0, steps, model_fn, c, dt, centre, f32abs):
    """All variants for one scene. pos0 centered. Returns dict of per-body-per-step error arrays (T,N) vs ref, plus ref min distance."""
    L = float(np.sqrt((pos0 ** 2).sum(-1).mean()))
    rng = np.random.default_rng(4738)
    direction = rng.normal(size=pos0.shape)
    ref, mind = sim(pos0, vel0, steps, torch.float64, c=c, dt=dt)
    out = {"L": L, "N": len(pos0), "mind_ref": np.minimum.accumulate(mind, 0)}
    for rel in RELS:
        pp, _ = sim(pos0 + rel * L * direction, vel0, steps, torch.float64, c=c, dt=dt)
        out[f"pert_{rel:g}"] = body_err(pp, ref)[1:]
    out["f32"] = body_err(sim(pos0, vel0, steps, torch.float32, c=c, dt=dt)[0], ref)[1:]
    if f32abs:
        out["f32abs"] = body_err(sim(pos0 + centre, vel0, steps, torch.float32, c=c, dt=dt)[0] - centre, ref)[1:]
    out["sub8"] = body_err(sim(pos0, vel0, steps, torch.float64, substeps=8, c=c, dt=dt)[0], ref)[1:]
    out["sub2"] = body_err(sim(pos0, vel0, steps, torch.float64, substeps=2, c=c, dt=dt)[0], ref)[1:]
    mp = model_fn(pos0, vel0)
    out["model"] = body_err(mp, ref)[1:]
    out["model_vs_sub8"] = None
    mp2 = model_fn(pos0 + 1e-5 * L * direction, vel0)
    out["model_selfpert_1e-05"] = body_err(mp2, mp)[1:]
    del out["model_vs_sub8"]
    return out, ref


def summarize(recs, keys, groups):
    """groups: name -> function(rec) -> bool mask over bodies (N,) per step handled by caller. Returns {group: {key: [err@k]}} (scene-mean of body-mean)."""
    res = {}
    for gname, gfn in groups.items():
        res[gname] = {}
        for key in keys:
            vals = []
            for k in KS:
                s = []
                for r in recs:
                    m = gfn(r, k)
                    if m.sum() > 0:
                        s.append(r[key][k - 1][m].mean())
                vals.append((float(np.mean(s)) if s else None, len(s)))
            res[gname][key] = vals
    return res


def lyap(curve_by_rel, dt):
    """Fit ln(mean err) vs t between err > 30x initial perturbation and err < 0.3 of final plateau; return rate per time unit per rel."""
    out = {}
    for rel, e in curve_by_rel.items():
        e = np.asarray(e)
        lo = e[0] * 30
        hi = 0.3 * e.max()
        idx = np.where((e > lo) & (e < hi))[0]
        if len(idx) < 8:
            out[rel] = None
            continue
        t = (idx + 1) * dt
        out[rel] = float(np.polyfit(t, np.log(e[idx]), 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bh", "expb"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--scenes", type=int, default=48)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    steps, dt = args.steps, args.dt
    recs, meta = [], []
    if args.mode == "bh":
        seeds = [int(s) for s in (args.seeds or "9100,9200,9300,4738").split(",")]
        model = load_model("E", args.ckpt, dt, dev)
        c = args.c
        scenes = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            n = args.n
            pos0 = np.clip(rng.normal(0.0, args.sigma, (n, 2)), -29.0, 29.0)
            vel0 = rng.normal(0.0, args.vfac * np.sqrt(n / (4 * args.sigma)), (n, 2))
            vel0 -= vel0.mean(0)
            scenes.append((seed, pos0, vel0))
        mfn = lambda p, v: model_rollout_bh(model, p, v, steps, dt, c, dev)
        f32abs = False
        centre = 0.0
        check = scenes[0]
        chk = gs.rollout_torch(check[1], check[2], steps, dev, dt=dt, substeps=4, eps=args.eps, relativistic=True, c=c)[0]
    else:
        seeds = [int(s) for s in (args.seeds or "9000,12000,9100,9200,9300").split(",")]
        model = load_model("B", args.ckpt, dt, dev)
        c = None
        cfg = tsf.EXPS["B"]
        scenes = []
        for seed in seeds:
            data = tsf.make_data(cfg, args.scenes, 1, seed, dev, dt, args.eps)
            for i, d in enumerate(data):
                scenes.append((seed * 1000 + i, d[0][0] - gs.CENTER, d[1][0]))
        mfn = lambda p, v: model_rollout_expb(model, p, v, steps, dev)
        f32abs = True
        centre = gs.CENTER
        check = scenes[0]
        chk = gs.rollout_torch(check[1] + gs.CENTER, check[2], steps, dev, dt=dt, substeps=4, eps=args.eps)[0] - gs.CENTER
    ref0, _ = sim(check[1], check[2], steps, torch.float64, c=c, dt=dt)
    print("sim reimplementation max |diff| vs gs.rollout_torch:", float(np.abs(ref0 - chk).max()), flush=True)
    assert np.abs(ref0 - chk).max() < 1e-7
    for i, (sid, p0, v0) in enumerate(scenes):
        r, _ = scene_record(sid, p0, v0, steps, mfn, c, dt, centre, f32abs)
        r["id"] = sid
        recs.append(r)
        if i % 10 == 0:
            print(f"scene {i}/{len(scenes)} id {sid} N {r['N']} model err@20/50/100 {[round(float(r['model'][k - 1].mean()), 4) for k in (20, 50, 100)]} "
                  f"pert1e-5 {[round(float(r['pert_1e-05'][k - 1].mean()), 5) for k in (20, 50, 100)]}", flush=True)

    keys = ["model", "model_selfpert_1e-05", "f32", "sub8", "sub2"] + [f"pert_{r:g}" for r in RELS] + (["f32abs"] if f32abs else [])
    groups = {"all": lambda r, k: np.ones(r["N"], bool),
              "close_by_k": lambda r, k: r["mind_ref"][k - 1] < RC,
              "noclose_by_k": lambda r, k: r["mind_ref"][k - 1] >= RC}
    if args.mode == "expb":
        groups.update({"N10-30": lambda r, k: np.full(r["N"], 10 <= r["N"] <= 30),
                       "N31-60": lambda r, k: np.full(r["N"], 31 <= r["N"] <= 60),
                       "N61-100": lambda r, k: np.full(r["N"], 61 <= r["N"] <= 100)})
    res = summarize(recs, keys, groups)
    allcurves = {k: np.mean([r[k].mean(1) for r in recs], 0) for k in keys}
    rate = lyap({r_: allcurves[f"pert_{r_:g}"] for r_ in RELS}, dt)
    fracclose = {str(k): float(np.mean([(r["mind_ref"][k - 1] < RC).mean() for r in recs])) for k in KS}
    per_scene_model = {str(r["id"]): [float(r["model"][k - 1].mean()) for k in KS] for r in recs}
    per_scene_p5 = {str(r["id"]): [float(r["pert_1e-05"][k - 1].mean()) for k in KS] for r in recs}
    out = {"mode": args.mode, "ks": KS, "n_scenes": len(recs), "seeds": seeds, "rels": RELS, "summary": res, "lyap_per_time_unit": {str(k): v for k, v in rate.items()},
           "frac_bodies_close_by_k": fracclose, "mean_curves": {k: v.tolist() for k, v in allcurves.items()},
           "per_scene_model": per_scene_model, "per_scene_pert1e-5": per_scene_p5, "ckpt": args.ckpt, "dt": dt}
    json.dump(out, open(f"results/{args.tag}.json", "w"))
    print("err@5/10/20/50/100 (all bodies):")
    for k in keys:
        print(f"{k:>22}", [None if v[0] is None else float(f"{v[0]:.3g}") for v in res["all"][k]])
    print("lyapunov rates per time unit:", rate)
    print("frac bodies with close encounter by k:", fracclose)


if __name__ == "__main__":
    main()
