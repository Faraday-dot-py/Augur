"""Fix 1 test: does a larger / adaptive pair radius remove missed fast collisions
in soup B? Head-on pair (offset 0 and 0.6) at speeds 3..80: pass-through (y order
flips), relative-velocity reversal ratio (truth -1 for offset 0), relKE ratio, momentum
error, for neighbor radius variants.

Usage: PYTHONPATH=. python3 scripts/fix1_probe.py --device cuda
"""
import argparse
import json

import numpy as np

import scripts.energy_probe2 as p2
from scripts.energy_fix_common import build_model

SPEEDS = [3, 6, 10, 13, 15, 20, 30, 40, 60, 80]
VARIANTS = {
    "base_r4": dict(adaptive_radius="off", neighbor_radius=4.0),
    "adaptive_pair": dict(adaptive_radius="pair", neighbor_radius=4.0),
    "adaptive_all": dict(adaptive_radius="all", neighbor_radius=4.0),
    "fixed_r16": dict(adaptive_radius="off", neighbor_radius=16.0),
    "fixed_r32": dict(adaptive_radius="off", neighbor_radius=32.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/fix1_probe.json")
    args = ap.parse_args()
    out = {}
    for name, kw in VARIANTS.items():
        model = build_model("soupb", args.checkpoint, 800, args.device, **kw)
        rows = []
        for off in (0.0, 0.6):
            for s in SPEEDS:
                p0 = [[400.0, 396.0], [400.0 + off, 404.0]]
                v0 = [[0.0, float(s)], [0.0, -float(s)]]
                steps = 30
                row = {"offset": off, "speed_in": s}
                for label, (ps, vs) in (("truth", p2.truth_states(p0, v0, 800, steps, args.device)),
                                        ("model", p2.model_states(model, p0, v0, steps, args.device))):
                    rel0, rel1 = vs[0][1] - vs[0][0], vs[-1][1] - vs[-1][0]
                    exp_x = 2 * p2.G * p2.DT * steps
                    row[label + "_passed_through"] = bool(ps[-1][1][1] < ps[-1][0][1])
                    row[label + "_relvy_ratio"] = float(rel1[1] / rel0[1])
                    row[label + "_relKE_ratio"] = float((rel1 ** 2).sum() / (rel0 ** 2).sum())
                    row[label + "_pmom_err"] = float(np.abs((vs[-1].sum(0) - vs[0].sum(0)) - np.array([exp_x, 0.0])).max())
                    row[label + "_finite"] = bool(np.isfinite(vs).all())
                rows.append(row)
                print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        out[name] = rows
    json.dump(out, open(args.out, "w"), default=float)
    print("done")


if __name__ == "__main__":
    main()
