"""Dump float64 PyTorch rollouts of a ScatterField checkpoint (free rollout, Verlet, relative
coordinates) to web/scatter/tests/vectors.json for web/scatter/tests/verify.mjs.

Usage:
    PYTHONPATH=.:/path/to/repo-with-scatter_field python3 scripts/export_web_testvectors_scatter.py --checkpoint checkpoints/budgetB/sfv/B_ms_kp_pot_v_g128.pt
"""
import argparse
import json

import numpy as np
import torch

from scripts.export_web_weights_scatter import CONFIG
from scripts.scatter_field import ScatterField


def scene(rng, n, spread=5.0, speed=0.5):
    f, fv = (n / 8) ** 0.5, (n / 8) ** 0.25
    pos = rng.uniform(-spread * f, spread * f, (n, 2))
    vel = rng.normal(0.0, speed * fv, (n, 2))
    vel -= vel.mean(0)
    return pos, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/budgetB/sfv/B_ms_kp_pot_v_g128.pt")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    model = ScatterField(grid=CONFIG["grid"], extent=CONFIG["extent"], net="unet", dt=CONFIG["dt"], kernel=True, pp=CONFIG["pp"],
                         split=True, potential=True, verlet=True).double()
    model.load_state_dict({k: v.double() for k, v in torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"].items()})
    model.eval()
    rng = np.random.default_rng(4738)
    out = {"steps": args.steps, "scenes": {}}
    for name, n in (("n8", 8), ("n30", 30), ("n100", 100)):
        p, v = scene(rng, n)
        pos, vel = torch.tensor(p)[None], torch.tensor(v)[None]
        mass, mask = torch.ones(1, n, dtype=torch.float64), torch.ones(1, n, dtype=torch.float64)
        field = model.init_field(1, "cpu", torch.float64)
        ps, vs = [pos[0].tolist()], [vel[0].tolist()]
        with torch.no_grad():
            for t in range(args.steps):
                pos, vel, field, _ = model.step(pos, vel, mass, mask, field)
                ps.append(pos[0].tolist())
                vs.append(vel[0].tolist())
        out["scenes"][name] = {"pos": ps, "vel": vs, "field0": field[0][0, 0].tolist() if name == "n8" else None}
        print(name, n, "bodies")
    with open("web/scatter/tests/vectors.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
