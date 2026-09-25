"""Strip-tiled vs global evaluation of one trained model (GPU).
(a) conservative-contact (cons_pure.pt), n=800 box, 6400 balls (sparse, density 0.01, and a floor pile): each step the balls are split
into x-strips with a halo, every strip runs model.step_free on its balls+halo, and only the strip's own balls are kept (hidden
state gathered/scattered with the balls). One-step exactness from shared states (no chaos), 20-step rollout divergence vs a global
rollout, with a random-perturbation noise floor. Sweeps strips and halo width (halo 4 = graph radius, 12 = 8 substeps x 1.5 contact
reach, 24 = plus displacement margin).
(b) CentralForceDynamics via scripts/gravity_1b.tiled_accel (halo = interaction radius, exact by construction), strips 1 vs S.

Usage: PYTHONPATH=. python scripts/tile_vs_global.py --out results/tile_vs_global.json
"""
import argparse
import json
import random

import torch

from model.central_force import CentralForceDynamics
from model.dataset import make_scenario_uniform
from scripts import gravity_1b as g1
from scripts.lut_bench import dense_state
from scripts.tile_box_test import make_cons

REP = (1, 5, 10, 20)


def tiled_step(model, pos, vel, hid, n, strips, halo):
    x = pos[:, 0]
    bounds = [-1e9] + [i * (n - 1.0) / strips for i in range(1, strips)] + [1e9]
    np_, nv, nh = pos.clone(), vel.clone(), hid.clone()
    for s in range(strips):
        a, b = bounds[s], bounds[s + 1]
        idx = ((x >= a - halo) & (x < b + halo)).nonzero().squeeze(1)
        core = (x[idx] >= a) & (x[idx] < b)
        if not core.any():
            continue
        p, v, h, _ = model.step_free(pos[idx], vel[idx], hid[idx], render=False)
        keep = idx[core]
        np_[keep], nv[keep], nh[keep] = p[core], v[core], h[core]
    return np_, nv, nh


def roll(model, pos, vel, steps, n=None, strips=1, halo=0.0, keep_at=REP):
    hid = torch.zeros(pos.shape[0], model.dynamics.hidden_dim, device=pos.device)
    out = {0: (pos.clone(), vel.clone())}
    with torch.no_grad():
        for t in range(1, steps + 1):
            if strips == 1:
                pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            else:
                pos, vel, hid = tiled_step(model, pos, vel, hid, n, strips, halo)
            if t in keep_at:
                out[t] = (pos.clone(), vel.clone())
    return out


def diff(a, b):
    d = (a - b).norm(dim=-1)
    return {"max": float(d.max()), "mean": float(d.mean()), "frac_gt_1e-3": float((d > 1e-3).float().mean())}


def cons_part(args, dev):
    n = 800
    out = {}
    nb = int(0.01 * n * n)
    balls = make_scenario_uniform(nb, n, 2.3, random.Random(9000))
    sp = torch.tensor([[b["x"], b["y"]] for b in balls], device=dev)
    sv = torch.tensor([[b["vx"], b["vy"]] for b in balls], device=dev)
    nd, pd, vd = dense_state(6000, dev, 4738)
    scen = {"sparse_6400": (n, sp, sv), f"pile_6000_n{nd}": (nd, pd, vd)}
    for name, (nn_, p0, v0) in scen.items():
        model = make_cons(args.checkpoint, nn_, p0.shape[0], dev)
        base = roll(model, p0, v0, 20)
        gen = torch.Generator(device=dev).manual_seed(1)
        noise = {}
        for mag in (1e-6, 3e-5):
            r = roll(model, p0 + (torch.rand(p0.shape, device=dev, generator=gen) * 2 - 1) * mag, v0, 20)
            noise[f"perturb_{mag:g}"] = {str(t): diff(r[t][0], base[t][0]) for t in REP}
        rows = {"noise": noise, "tiled": {}}
        st = {t: base[t] for t in (0, 5, 10)}
        for strips in args.strips:
            for halo in args.halos:
                key = f"S{strips}_h{halo:g}"
                one = {}
                for t0 in (0, 5, 10):
                    p, v = st[t0]
                    hid = torch.zeros(p.shape[0], model.dynamics.hidden_dim, device=dev)
                    with torch.no_grad():
                        gp, gv, _, _ = model.step_free(p, v, hid, render=False)
                        tp, tv, _ = tiled_step(model, p, v, hid, nn_, strips, halo)
                    one[str(t0)] = {"dp": diff(tp, gp), "dv": diff(tv, gv)}
                r = roll(model, p0, v0, 20, n=nn_, strips=strips, halo=halo)
                rows["tiled"][key] = {"one_step": one, "rollout": {str(t): diff(r[t][0], base[t][0]) for t in REP}}
                print(name, key, "one_step_t0", json.dumps(one["0"]), "rollout20", json.dumps(rows["tiled"][key]["rollout"]["20"]), flush=True)
        out[name] = rows
        print(name, "noise", json.dumps({k: v["20"] for k, v in noise.items()}), flush=True)
        json.dump(out, open(args.out + ".part", "w"))
    return out


def central_run(dyn, radius, pos, vel, steps, strips, keep_at):
    force = g1.model_force(dyn)
    out = {0: pos.clone()}
    for t in range(1, steps + 1):
        pos, vel = g1.step(force, pos, vel, 0.1, radius, strips)
        if t in keep_at:
            out[t] = pos.clone()
    return out


def central_part(args, dev):
    dyn = CentralForceDynamics(dt=0.1).to(dev)
    dyn.load_state_dict(torch.load(args.central_checkpoint, map_location=dev))
    rep = (1, 5, 20, 100)
    out = {}
    for N, radius in args.central_cases:
        pos, vel, _ = g1.init_state(N, dev, 4738)
        base = central_run(dyn, radius, pos, vel, 100, 1, rep)
        gen = torch.Generator(device=dev).manual_seed(1)
        noise = central_run(dyn, radius, pos + (torch.rand(pos.shape, device=dev, generator=gen) * 2 - 1) * 6e-5, vel, 100, 1, rep)
        rows = {"noise_perturb_6e-5": {str(t): diff(noise[t], base[t]) for t in rep}}
        for strips in args.central_strips:
            r = central_run(dyn, radius, pos, vel, 100, strips, rep)
            rows[f"S{strips}"] = {str(t): diff(r[t], base[t]) for t in rep}
            print("central", N, radius, strips, json.dumps(rows[f"S{strips}"]["1"]), json.dumps(rows[f"S{strips}"]["100"]), flush=True)
        print("central", N, radius, "noise", json.dumps(rows["noise_perturb_6e-5"]["100"]), flush=True)
        out[f"N{N}_r{radius:g}"] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cons_pure.pt")
    ap.add_argument("--central-checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--strips", type=int, nargs="+", default=[2, 8, 32])
    ap.add_argument("--halos", type=float, nargs="+", default=[4.0, 12.0, 24.0])
    ap.add_argument("--central-strips", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--out", default="results/tile_vs_global.json")
    args = ap.parse_args()
    args.central_cases = [(8000, 4.0), (64000, 4.0), (8000, 100.0)]
    dev = torch.device("cuda")
    out = {"cons": cons_part(args, dev)}
    json.dump(out, open(args.out, "w"))
    out["central"] = central_part(args, dev)
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
