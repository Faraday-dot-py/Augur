"""Dump PyTorch rollouts of a CentralForceDynamics checkpoint (no walls, no
box, free rollout) plus a float64 one-step reference to
web/gravity/tests/vectors.json for web/gravity/tests/verify.mjs.

Usage:
    PYTHONPATH=. python3 scripts/export_web_testvectors_central.py --checkpoint checkpoints/gravity_central_v1.pt
"""
import argparse
import json

import numpy as np
import torch

from model.central_force import CentralForceDynamics


def build(checkpoint, dtype, dt, neighbor_radius):
    dyn = CentralForceDynamics(dt=dt, neighbor_radius=neighbor_radius).to(dtype)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model", state)
    dyn.load_state_dict({k: v.to(dtype) for k, v in state.items()})
    dyn.eval()
    return dyn


def scene(rng, n, spread, speed):
    f, fv = (n / 8) ** 0.5, (n / 8) ** 0.25
    pos = rng.uniform(-spread * f, spread * f, (n, 2))
    vel = rng.normal(0.0, speed * fv, (n, 2))
    vel -= vel.mean(0)
    return torch.tensor(pos, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)


def rollout(dyn, pos, vel, steps, dt):
    hidden = torch.zeros(pos.shape[0], dyn.hidden_dim, dtype=pos.dtype)
    states = [(pos, vel)]
    with torch.no_grad():
        for _ in range(steps):
            dp, dv, hidden = dyn(pos, vel, hidden)
            pos = pos + vel * dt + dp
            vel = vel + dv
            states.append((pos, vel))
    return states


def one_step_f64(dyn64, states, dt):
    out = []
    with torch.no_grad():
        for pos, vel in states[:-1]:
            hidden = torch.zeros(pos.shape[0], 1, dtype=torch.float64)
            dp, dv, _ = dyn64(pos.double(), vel.double(), hidden)
            out.append((pos.double() + vel.double() * dt + dp, vel.double() + dv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--neighbor-radius", type=float, default=100.0)
    args = ap.parse_args()

    dyn = build(args.checkpoint, torch.float32, args.dt, args.neighbor_radius)
    dyn64 = build(args.checkpoint, torch.float64, args.dt, args.neighbor_radius)
    rng = np.random.default_rng(4738)
    scenes = {
        "small8": (8, 5.0, 0.5, 60),
        "mid50": (50, 5.0, 0.5, 60),
        "cluster200": (200, 5.0, 0.5, 40),
        "wide12": (12, 40.0, 0.2, 60),
        "fast30": (30, 5.0, 2.0, 40),
    }
    out = {"dt": args.dt, "neighbor_radius": args.neighbor_radius, "scenes": {}}
    for name, (count, spread, speed, steps) in scenes.items():
        pos, vel = scene(rng, count, spread, speed)
        states = rollout(dyn, pos, vel, steps, args.dt)
        entry = {
            "steps": steps,
            "pos": [s[0].tolist() for s in states],
            "vel": [s[1].tolist() for s in states],
        }
        exact = one_step_f64(dyn64, states, args.dt)
        entry["pos64"] = [p.tolist() for p, _ in exact]
        entry["vel64"] = [v.tolist() for _, v in exact]
        out["scenes"][name] = entry
        print(name, count, "bodies,", steps, "steps")
    with open("web/gravity/tests/vectors.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
