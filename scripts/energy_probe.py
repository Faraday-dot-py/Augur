"""Energy-conservation probe for the token free-rollout model (soup B).

Per config (num balls, box n, gravity): truth vs free rollout energy
(KE - g*x, gravity along +x) per ball vs step; teacher-forced one-step energy
error by class (free flight / wall / pair) and speed bin, hidden zero vs carried;
single-ball speed sweep.

Usage: PYTHONPATH=. python3 scripts/energy_probe.py --out results/energy_probe.json
"""
import argparse
import json
import random

import numpy as np
import torch

from model.dataset import make_scenario_uniform
from scripts.ball_1k_rollout import truth
from scripts.eval_free_rollout import load_model

CONFIGS = [(4, 20, 9.0), (4, 20, 0.0), (4, 317, 9.0), (100, 100, 9.0), (100, 100, 0.0),
           (1000, 317, 9.0), (1000, 317, 1.0), (1000, 317, 0.0)]
REPORT = [0, 1, 5, 10, 20, 30, 50, 100, 150, 200]
SPEED_BINS = [0, 5, 10, 15, 20, 30, 50, 100]
DT, RADIUS = 0.15, 0.75


def energy(pos, vel, g):
    return 0.5 * (vel ** 2).sum(-1) - g * pos[..., 0]


def classify(pos, n):
    """0 free, 1 wall (within 1.5 of a wall), 2 pair (nearest ball < 3.0); pair wins."""
    lo = pos.min(dim=1).values
    hi = (n - 1 - pos).min(dim=1).values
    wall = torch.minimum(lo, hi) < 1.5
    d = torch.cdist(pos, pos) + torch.eye(pos.shape[0], device=pos.device) * 1e6
    pair = d.min(dim=1).values < 3.0 if pos.shape[0] > 1 else torch.zeros_like(wall)
    cls = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    cls[wall] = 1
    cls[pair] = 2
    return cls


def run_config(nb, n, g, seed, steps, ckpt, device):
    balls = make_scenario_uniform(nb, n, 2.3, random.Random(seed))
    tp, tv = truth(balls, n, steps, DT, g, RADIUS, 400.0, 8, device)
    model = load_model(ckpt, "free", n, 32, 4.0, False, True, True, True, True, True, True).to(device)
    model.dynamics.cell_graph = nb > 50
    hd = model.dynamics.hidden_dim
    tpt, tvt = torch.from_numpy(tp).to(device), torch.from_numpy(tv).to(device)

    pos, vel, hid = tpt[0].clone(), tvt[0].clone(), torch.zeros(nb, hd, device=device)
    free = {"tE": [float(energy(tpt[t], tvt[t], g).mean()) for t in range(steps + 1)], "tKE": [float((0.5 * (tvt[t] ** 2).sum(-1)).mean()) for t in range(steps + 1)], "KE": [], "E": [], "speed": [], "maxspeed": [], "err": [], "hid": [], "finite_step": None}
    with torch.no_grad():
        for t in range(steps + 1):
            ok = bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()) and float(pos.abs().max()) < 1e5
            if not ok:
                free["finite_step"] = t
                break
            free["KE"].append(float((0.5 * (vel ** 2).sum(-1)).mean()))
            free["E"].append(float(energy(pos, vel, g).mean()))
            sp = vel.norm(dim=1)
            free["speed"].append(float(sp.mean()))
            free["maxspeed"].append(float(sp.max()))
            free["err"].append(float((pos - tpt[t]).norm(dim=1).mean()))
            free["hid"].append(float(hid.abs().mean()))
            if t < steps:
                pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)

    tf = {}
    with torch.no_grad():
        hid_c = torch.zeros(nb, hd, device=device)
        for t in range(steps):
            p, v = tpt[t], tvt[t]
            cls = classify(p, n)
            sp = v.norm(dim=1)
            e_t = energy(p, v, g)
            dE_true = energy(tpt[t + 1], tvt[t + 1], g) - e_t
            for mode in ("zero", "carried"):
                h_in = torch.zeros(nb, hd, device=device) if mode == "zero" else hid_c
                np_, nv, nh, _ = model.step_free(p, v, h_in, render=False)
                if mode == "carried":
                    hid_c = nh
                dE = energy(np_, nv, g) - e_t
                dv_err = (nv - tvt[t + 1]).norm(dim=1)
                dvx_err = (nv[:, 0] - tvt[t + 1][:, 0])
                for c in range(3):
                    for b in range(len(SPEED_BINS) - 1):
                        m = (cls == c) & (sp >= SPEED_BINS[b]) & (sp < SPEED_BINS[b + 1]) & torch.isfinite(dE)
                        if int(m.sum()) == 0:
                            continue
                        key = f"{mode}|cls{c}|sp{SPEED_BINS[b]}"
                        a = tf.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0])
                        a[0] += int(m.sum())
                        a[1] += float((dE[m] - dE_true[m]).sum())
                        a[2] += float(dE_true[m].sum())
                        a[3] += float(dv_err[m].sum())
                        a[4] += float(dvx_err[m].sum())
    return free, tf


def speed_sweep(ckpt, device, n=317):
    model = load_model(ckpt, "free", n, 32, 4.0, False, True, True, True, True, True, True).to(device)
    hd = model.dynamics.hidden_dim
    out = {}
    with torch.no_grad():
        for g in (0.0, 9.0):
            rows = []
            for s in [1, 3, 6, 10, 15, 20, 30, 40, 60, 80]:
                for ax in ("x", "y"):
                    pos = torch.tensor([[150.0, 150.0]], device=device)
                    vel = torch.zeros(1, 2, device=device)
                    vel[0, 0 if ax == "x" else 1] = float(s)
                    hid = torch.zeros(1, hd, device=device)
                    np_, nv, nh, _ = model.step_free(pos, vel, hid, render=False)
                    dv1 = (nv - vel)[0].tolist()
                    p, v, h = pos.clone(), vel.clone(), hid
                    speeds = []
                    for _ in range(40):
                        p, v, h, _ = model.step_free(p, v, h, render=False)
                        speeds.append(float(v.norm()))
                    rows.append({"ax": ax, "speed0": s, "dv_step1_zero_hidden": dv1,
                                 "speed_after_40_freeflight": speeds[-1], "speed_at_10": speeds[9],
                                 "hid_abs_after_40": float(h.abs().mean())})
            out[f"g_model_only_{g}"] = rows
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/energy_probe.json")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    result = {"configs": {}, "speed_sweep": speed_sweep(args.checkpoint, args.device)}
    for nb, n, g in CONFIGS:
        if args.only and f"nb{nb}_n{n}_g{g}" not in args.only.split(","):
            continue
        seeds = [4738 + 1000 * i for i in range(4 if nb <= 100 else 1)]
        runs = []
        for s in seeds:
            free, tf = run_config(nb, n, g, s, args.steps, args.checkpoint, args.device)
            runs.append({"seed": s, "free": free, "tf": tf})
            print(f"done nb={nb} n={n} g={g} seed={s} finite_step={free['finite_step']}", flush=True)
        result["configs"][f"nb{nb}_n{n}_g{g}"] = runs
        json.dump(result, open(args.out, "w"))

        L = min(len(r["free"]["E"]) for r in runs)
        print(f"== nb={nb} n={n} g={g}: step  E_model(mean over seeds)  KE_model  speed  maxspeed  err  |hid|")
        for t in REPORT:
            if t < L:
                m = lambda k: float(np.mean([r["free"][k][t] for r in runs]))
                print(f"{t:4d} tE={m('tE'):9.2f} tKE={m('tKE'):9.2f} E={m('E'):10.2f} KE={m('KE'):9.2f} sp={m('speed'):7.2f} max={m('maxspeed'):7.2f} err={m('err'):8.3f} hid={m('hid'):.3f}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
