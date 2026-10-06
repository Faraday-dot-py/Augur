"""Dump PyTorch rollouts of a conservative-contact checkpoint (contain_state as
in scripts/realtime_sim.py) plus a per-substep trace of one ball to
web/token/tests/vectors.json for web/token/tests/verify.mjs.

Usage:
    PYTHONPATH=. python3 scripts/export_web_testvectors.py --checkpoint results/cons_pure.pt
"""
import argparse
import json

import numpy as np
import torch

from scripts.energy_fix_common import build_model
from scripts.realtime_sim import contain_state

MAX_SPEED = 30.0


def scene(rng, count, lo, hi, speed):
    pos = rng.uniform(lo, hi, (count, 2))
    vel = rng.uniform(-speed, speed, (count, 2))
    return torch.tensor(pos, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)


def rollout(model, pos, vel, steps, n):
    hidden = torch.zeros((pos.shape[0], model.dynamics.hidden_dim))
    states = [(pos, vel, hidden)]
    with torch.no_grad():
        for _ in range(steps):
            pos, vel, hidden, _ = model.step_free(pos, vel, hidden, render=False)
            pos, vel, hidden = contain_state(pos, vel, hidden, n, MAX_SPEED)
            states.append((pos, vel, hidden))
    return states


def one_step_f64(model64, states, n):
    out = []
    with torch.no_grad():
        for pos, vel, hidden in states[:-1]:
            p, v, _, _ = model64.step_free(pos.double(), vel.double(), hidden.double(), render=False)
            p, v, _ = contain_state(p, v, hidden.double(), n, MAX_SPEED)
            out.append((p, v))
    return out


def substep_trace(dyn, pos, vel, idx):
    h = dyn.dt / dyn.contact_substeps
    p, v = pos, vel
    a = dyn.contact_accel(p)
    out = {"pos": [p[idx].tolist()], "vel": [v[idx].tolist()], "acc": [a[idx].tolist()]}
    for _ in range(dyn.contact_substeps):
        v_half = v + 0.5 * h * a
        p = p + h * v_half
        a = dyn.contact_accel(p)
        v = v_half + 0.5 * h * a
        out["pos"].append(p[idx].tolist())
        out["vel"].append(v[idx].tolist())
        out["acc"].append(a[idx].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cons_pure.pt")
    args = ap.parse_args()
    n = 100
    model = build_model("cons", args.checkpoint, n, "cpu")
    model64 = build_model("cons", args.checkpoint, n, "cpu").double()
    rng = np.random.default_rng(4738)
    scenes = {
        "spread12": (12, 5.0, 95.0, 2.3, 60),
        "cluster30": (30, 40.0, 60.0, 2.3, 60),
        "dense100": (100, 30.0, 70.0, 2.3, 60),
        "fast_walls": (20, 2.0, 98.0, 25.0, 60),
        "contact_pack": (40, 45.0, 55.0, 4.0, 60),
    }
    out = {"n": n, "max_speed": MAX_SPEED, "scenes": {}}
    for name, (count, lo, hi, speed, steps) in scenes.items():
        pos, vel = scene(rng, count, lo, hi, speed)
        states = rollout(model, pos, vel, steps, n)
        entry = {
            "steps": steps,
            "pos": [s[0].tolist() for s in states],
            "vel": [s[1].tolist() for s in states],
            "hidden": [s[2].tolist() for s in states],
            "count": [s[0].shape[0] for s in states],
        }
        exact = one_step_f64(model64, states, n)
        entry["pos64"] = [p.tolist() for p, _ in exact]
        entry["vel64"] = [v.tolist() for _, v in exact]
        if name == "contact_pack":
            with torch.no_grad():
                d = torch.cdist(states[5][0], states[5][0]) + torch.eye(count) * 1e9
                idx = int(d.min(dim=1).values.argmin())
                entry["trace"] = {"t": 5, "idx": idx, **substep_trace(model64.dynamics, states[5][0].double(), states[5][1].double(), idx)}
        out["scenes"][name] = entry
        print(name, count, "->", states[-1][0].shape[0], "balls at end")
    with open("web/token/tests/vectors.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
