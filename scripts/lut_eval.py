"""Accuracy of LUT / adaptive-substep variants vs the unmodified conservative-contact model (GPU).
(a) table error vs MLP, (b) one-step max diff on sparse / dense / mixed states, (c) err@k, wall restitution, pair
head-on relKE (scripts/eval_energy_fix.py machinery, truth cached across variants), (d) 500-step rollouts of 1000 balls
(sparse ball_1k-style and a floor pile): E/ball, KE/ball, contact fraction, divergence from base and from truth.

Usage: PYTHONPATH=. python scripts/lut_eval.py --out results/lut_eval.json
"""
import argparse
import json
import random

import numpy as np
import torch

import scripts.energy_probe2 as p2
import scripts.eval_energy_fix as ef
from model.dataset import make_scenario_uniform
from scripts.energy_fix_common import build_model as _build
from scripts.energy_probe import energy
from scripts.lut_adaptive import install, parse
from scripts.lut_bench import contact_fraction, dense_state

G, DT = 9.0, 0.15
REPORT = [1, 5, 10, 20, 50, 100, 200, 500]
CURRENT = {"spec": "base"}


def cached(fn, keyfn):
    store = {}

    def wrapped(*a, **k):
        key = keyfn(*a, **k)
        if key not in store:
            store[key] = fn(*a, **k)
        return store[key]
    return wrapped


ef.truth = cached(ef.truth, lambda balls, n, steps, *r: (len(balls), n, steps, balls[0]["x"], balls[0]["y"], balls[-1]["x"]))
p2.truth_states = cached(p2.truth_states, lambda p0, v0, n, steps, device, g=G: (repr(p0), repr(v0), n, steps))


def build(kind, ckpt, n, device, **kw):
    model = _build(kind, ckpt, n, device, **kw)
    install(model, CURRENT["spec"])
    return model


ef.build_model = build


def table_error(model, spec):
    lut, cubic, _ = parse(spec)
    if not lut:
        return None
    fc = install(model, spec)
    dyn = model.dynamics
    out = {}
    with torch.no_grad():
        for name, hi, fn, tab in (("pair", 1.0, lambda x: x * dyn.pair_force(x[:, None])[:, 0] * dyn.force_scale, fc.pair_t),
                                  ("wall", 2.0, lambda x: x * dyn.wall_force(x[:, None])[:, 0] * dyn.force_scale, fc.wall_t)):
            x = torch.rand(200000, device=x_dev(dyn)) * hi
            ref, got = fn(x), tab(x)
            out[name] = {"max_abs": float((ref - got).abs().max()), "max_rel_of_peak": float((ref - got).abs().max() / ref.abs().max()),
                         "ref_peak": float(ref.abs().max())}
    install(model, "base")
    return out


def x_dev(dyn):
    return dyn.gravity.device


def states(device):
    n = 317
    balls = make_scenario_uniform(1000, n, 2.3, random.Random(4738))
    sp = torch.tensor([[b["x"], b["y"]] for b in balls], device=device)
    sv = torch.tensor([[b["vx"], b["vy"]] for b in balls], device=device)
    return {"sparse_t0": (n, sp, sv)}


def advance(model, pos, vel, k):
    hid = torch.zeros(pos.shape[0], model.dynamics.hidden_dim, device=pos.device)
    with torch.no_grad():
        for _ in range(k):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
    return pos, vel


def one_step(specs, ckpt, device):
    n = 317
    m = build("cons", ckpt, n, device)
    m.dynamics.cell_graph = False
    balls = make_scenario_uniform(1000, n, 2.3, random.Random(4738))
    p0 = torch.tensor([[b["x"], b["y"]] for b in balls], device=device)
    v0 = torch.tensor([[b["vx"], b["vy"]] for b in balls], device=device)
    nd, pd, vd = dense_state(1000, device, 4738)
    st = {"sparse_t0": (n, p0, v0)}
    md = build("cons", ckpt, nd, device)
    md.dynamics.cell_graph = False
    st["mixed_t100"] = (n,) + advance(m, p0, v0, 100)
    st["dense_t20"] = (nd,) + advance(md, pd, vd, 20)
    out = {}
    for name, (nn_, p, v) in st.items():
        model = m if nn_ == n else md
        cf, _ = contact_fraction(p)
        install(model, "base")
        with torch.no_grad():
            hid = torch.zeros(p.shape[0], 1, device=device)
            bp, bv, _, _ = model.step_free(p, v, hid, render=False)
        out[name] = {"contact_frac": cf, "base_dv_median": float((bv - v).norm(dim=1).median()), "variants": {}}
        for spec in specs:
            fc = install(model, spec)
            with torch.no_grad():
                sp_, sv_, _, _ = model.step_free(p, v, hid, render=False)
            r = {"max_dp": float((sp_ - bp).norm(dim=1).max()), "max_dv": float((sv_ - bv).norm(dim=1).max()),
                 "mean_dv": float((sv_ - bv).norm(dim=1).mean())}
            if fc is not None and fc.stats["active"]:
                r["active_frac"] = fc.stats["active"][-1] / fc.stats["n"][-1]
            out[name]["variants"][spec] = r
        install(model, "base")
    return out


def rollout(model, pos, vel, steps):
    hid = torch.zeros(pos.shape[0], model.dynamics.hidden_dim, device=pos.device)
    ps, vs = [pos], [vel]
    with torch.no_grad():
        for _ in range(steps):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            ps.append(pos)
            vs.append(vel)
    return torch.stack(ps), torch.stack(vs)


def long_rollouts(specs, ckpt, device, steps=500):
    out = {}
    n = 317
    balls = make_scenario_uniform(1000, n, 2.3, random.Random(4738))
    nd, pd, vd = dense_state(1000, device, 4738)
    dballs = [{"x": float(a), "y": float(b), "vx": 0.0, "vy": 0.0} for a, b in pd.cpu().tolist()]
    scen = {"sparse_1k": (n, balls), "pile_1k": (nd, dballs)}
    for name, (nn_, bl) in scen.items():
        tp, tv = ef.truth(bl, nn_, steps, DT, G, 0.75, 400.0, 8, device)
        tp, tv = torch.from_numpy(tp).to(device), torch.from_numpy(tv).to(device)
        p0, v0 = tp[0].clone(), tv[0].clone()
        model = build("cons", ckpt, nn_, device)
        model.dynamics.cell_graph = False
        base = None
        rows = {}
        for spec in ["base"] + [s for s in specs if s != "base"]:
            fc = install(model, spec)
            ps, vs = rollout(model, p0, v0, steps)
            if spec == "base":
                base = (ps, vs)
            r = {"finite": bool(torch.isfinite(ps).all() and torch.isfinite(vs).all())}
            e = energy(ps, vs, G).mean(1)
            ke = 0.5 * (vs ** 2).sum(-1).mean(1)
            r["E"] = {str(t): float(e[t]) for t in REPORT if t < len(e)}
            r["KE"] = {str(t): float(ke[t]) for t in REPORT if t < len(e)}
            r["mean_x"] = {str(t): float(ps[t][:, 0].mean()) for t in REPORT}
            r["div_vs_base"] = {str(t): float((ps[t] - base[0][t]).norm(dim=1).mean()) for t in REPORT}
            r["err_vs_truth"] = {str(t): float((ps[t] - tp[t]).norm(dim=1).mean()) for t in REPORT}
            r["contact_frac"] = {str(t): contact_fraction(ps[t])[0] for t in (100, 500)}
            r["Edrift_100_500"] = float(e[500] - e[100])
            if fc is not None and fc.stats["active"]:
                r["active_frac_mean"] = sum(fc.stats["active"]) / sum(fc.stats["n"])
            rows[spec] = r
            print(name, spec, json.dumps({k: r[k] for k in ("finite", "E", "div_vs_base")}), flush=True)
        et = energy(tp, tv, G).mean(1)
        rows["truth"] = {"E": {str(t): float(et[t]) for t in REPORT}, "KE": {str(t): float((0.5 * (tv[t] ** 2).sum(-1)).mean()) for t in REPORT},
                         "contact_frac": {str(t): contact_fraction(tp[t])[0] for t in (100, 500)}}
        install(model, "base")
        out[name] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cons_pure.pt")
    ap.add_argument("--variants", default="lut256,lut1024,lut4096,lut1024c,lut4096c,skin0,skin1,skin2,lut1024+skin1,lut4096c+skin1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[9000, 12000])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/lut_eval.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    specs = args.variants.split(",")
    out = {"table_error": {}, "one_step": {}, "long": {}, "eval": {}}
    m = _build("cons", args.checkpoint, 317, dev)
    for s in specs:
        te = table_error(m, s)
        if te:
            out["table_error"][s] = te
    print("table_error", json.dumps(out["table_error"]), flush=True)
    out["one_step"] = one_step(specs, args.checkpoint, dev)
    print("one_step", json.dumps(out["one_step"]), flush=True)
    json.dump(out, open(args.out, "w"), default=float)
    out["long"] = long_rollouts(specs, args.checkpoint, dev)
    json.dump(out, open(args.out, "w"), default=float)
    for spec in ["base"] + specs:
        CURRENT["spec"] = spec
        r = {"n20_b4": ef.rollout_metrics("cons", args.checkpoint, {}, dev, args.seeds, 4, 20),
             "n100_b100": ef.rollout_metrics("cons", args.checkpoint, {}, dev, args.seeds, 100, 100, per_seed=3)}
        r.update(ef.wall_pair("cons", args.checkpoint, {}, dev))
        out["eval"][spec] = r
        print("EVAL", spec, json.dumps(r["n20_b4"]["err_mean"]), json.dumps(r["n100_b100"]["err_mean"]), flush=True)
        json.dump(out, open(args.out, "w"), default=float)
    print("done")


if __name__ == "__main__":
    main()
