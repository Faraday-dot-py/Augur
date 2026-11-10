"""Fail-loudly check for the architecture-push stage 1 jobs: the control loads under the shipped scatter_field.py and takes one finite step on a scene."""
import argparse

import numpy as np
import torch

from scripts import scatter_scaling as ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="kern/E_frozen.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = ss.load(args.control, dev)
    pos0, vel0 = ss.make_scene(200, 8.0, 0.3, 64.0, 4738, 0)
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    mass, mask = torch.ones(1, 200, device=dev), torch.ones(1, 200, device=dev)
    with torch.no_grad():
        p, v, _, _ = model.step(pos, vel, mass, mask, model.init_field(1, dev))
    if not (torch.isfinite(p).all() and torch.isfinite(v).all()):
        raise SystemExit("control step not finite")
    moved = float((p - pos).norm(dim=-1).mean())
    print(f"control ok: grid {model.grid} extent {model.extent} mean move {moved:.4f} np {np.__version__} torch {torch.__version__}")
    if not 0.0 < moved < 5.0:
        raise SystemExit(f"control mean move {moved} implausible")


if __name__ == "__main__":
    main()
