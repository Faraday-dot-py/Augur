"""Component ablation of soup B on the controlled wall / pair contact curves
(scripts/energy_probe2.py): which sub-module (softmax attention message, pair
head, wall head) produces the energy loss and momentum error.

Usage: PYTHONPATH=. python3 scripts/energy_probe3.py
"""
import json

import numpy as np
import torch

import scripts.energy_probe2 as p2
from scripts.eval_free_rollout import load_model

p2.SPEEDS = [3, 6, 10, 15, 20, 30, 40]


def build(ckpt, device, variant):
    model = load_model(ckpt, "free", 800, 32, 4.0, False, True, True, True, True, True, True).to(device)
    model.dynamics.cell_graph = False
    dyn = model.dynamics
    if "noattn" in variant:
        dyn.gru.register_forward_pre_hook(lambda m, a: (a[0] * 0.0, a[1]))
    if "nopair" in variant:
        dyn.pair_head = None
    if "nowall" in variant:
        dyn.wall_head = None
    return model


def pair_detail(model, device, n=800):
    rows = []
    for off in (0.0, 0.6):
        for s in (3, 10, 20, 40):
            p0 = [[400.0, 396.0], [400.0 + off, 404.0]]
            v0 = [[0.0, float(s)], [0.0, -float(s)]]
            steps = 30
            tps, tvs = p2.truth_states(p0, v0, n, steps, device)
            mps, mvs = p2.model_states(model, p0, v0, steps, device)
            exp_x = 2 * p2.G * p2.DT * steps
            row = {"offset": off, "speed_in": s}
            for label, vs in (("truth", tvs), ("model", mvs)):
                dp = vs[-1].sum(0) - vs[0].sum(0)
                row[label + "_dPx_minus_grav"] = float(dp[0] - exp_x)
                row[label + "_dPy"] = float(dp[1])
                rel0 = vs[0][1] - vs[0][0]
                rel1 = vs[-1][1] - vs[-1][0]
                row[label + "_relvy_ratio"] = float(rel1[1] / rel0[1])
            rows.append(row)
    return rows


def main():
    ckpt = "checkpoints/token_model_soup_b.pt"
    dev = "cuda"
    out = {}
    for variant in ("full", "noattn", "nopair", "noattn_nopair", "nowall"):
        model = build(ckpt, dev, variant)
        wc = p2.wall_curve(model, dev)
        pc = pair_detail(model, dev)
        out[variant] = {"wall": wc, "pair": pc}
        print("== variant", variant, flush=True)
        for r in wc:
            if r["wall"] == "y_wall":
                print(f"  y_wall s={r['speed_in']:3d} restitution model {r['model_ratio']:.3f} (truth {r['truth_ratio']:.3f})", flush=True)
        for r in wc:
            if r["wall"] == "x_floor":
                print(f"  x_floor s={r['speed_in']:3d} dE model {r['model_dE']:9.2f} (truth {r['truth_dE']:.2f})", flush=True)
        for r in pc:
            print(f"  pair off={r['offset']} s={r['speed_in']:3d} dPy {r['model_dPy']:8.3f} dPx-grav {r['model_dPx_minus_grav']:8.3f} relvy_ratio {r['model_relvy_ratio']:.3f} (truth {r['truth_relvy_ratio']:.3f})", flush=True)
    json.dump(out, open("results/energy_probe3.json", "w"), default=float)
    print("done")


if __name__ == "__main__":
    main()
