"""Roll the trained LJForceDynamics out from the initial state of each truth scene (same ramp/thermostat),
and re-run the truth from the same float32 initial state for the first frames as a chaos noise floor."""
import argparse

import numpy as np
import torch

from model.lj_force import LJForceDynamics
from scripts import lj_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/lj_force.pt")
    ap.add_argument("--scenes", default=",".join(lj_sim.SCENES))
    ap.add_argument("--floor-frames", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prefix", default="results/lj")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=dev)
    model = LJForceDynamics(dt=ck["args"]["dt"], substeps=20, width=ck["args"].get("width", 64)).double().to(dev)
    model.load_state_dict(ck["state"])
    model.requires_grad_(False)
    model.build_table()
    for name in args.scenes.split(","):
        truth = np.load(f"{args.prefix}_{name}.npz")
        dt = float(truth["dt"])
        rec = int(round(truth["stats"][1, 0] - truth["stats"][0, 0]))
        steps = int(truth["stats"][-1, 0])
        pos, vel, box = truth["pos"][0].astype(np.float64), truth["vel"][0].astype(np.float64), truth["box"]
        ramp = lj_sim.SCENES[name].get("ramp")
        fp, fv, st = lj_sim.run(pos, vel, box, steps, rec, dt, dev, ramp=ramp, force_fn=model.force_fn)
        np.savez_compressed(f"{args.prefix}_model{args.tag}_{name}.npz", pos=fp, vel=fv, box=box, stats=st, dt=dt, rho=truth["rho"])
        fp, fv, st = lj_sim.run(pos, vel, box, args.floor_frames * rec, rec, dt, dev, ramp=ramp)
        np.savez_compressed(f"{args.prefix}_rerun{args.tag}_{name}.npz", pos=fp, vel=fv, box=box, stats=st, dt=dt, rho=truth["rho"])
        print(name, "model E/N", st[0, 3], "T last model rollout done", flush=True)


if __name__ == "__main__":
    main()
