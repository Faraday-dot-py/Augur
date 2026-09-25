"""Strip-tiled vs global conservative-contact model at speeds up to 40 (training cap 40): n=800, uniform balls, per-component speed
+-vmax, densities 0.01 and 0.03, strips 8/32 x halo 4/12/24/48. One step from shared states (t=0, t=5) and 20-step rollout diff vs
global, with a 1e-6 input-perturbation noise floor. Same tiling as scripts/tile_vs_global.py.

Usage: PYTHONPATH=. python scripts/tile_speed_test.py --out results/tile_speed_test.json
"""
import argparse
import json
import random

import torch

from model.dataset import make_scenario_uniform
from scripts.tile_box_test import make_cons
from scripts.tile_vs_global import REP, diff, roll, tiled_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cons_pure.pt")
    ap.add_argument("--speeds", type=float, nargs="+", default=[2.3, 10.0, 20.0, 40.0])
    ap.add_argument("--densities", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--strips", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--halos", type=float, nargs="+", default=[4.0, 12.0, 24.0, 48.0])
    ap.add_argument("--out", default="results/tile_speed_test.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    n = 800
    out = {}
    for dens in args.densities:
        nb = int(dens * n * n)
        model = make_cons(args.checkpoint, n, nb, dev)
        for vmax in args.speeds:
            balls = make_scenario_uniform(nb, n, vmax, random.Random(9000))
            p0 = torch.tensor([[b["x"], b["y"]] for b in balls], device=dev)
            v0 = torch.tensor([[b["vx"], b["vy"]] for b in balls], device=dev)
            base = roll(model, p0, v0, 20, keep_at=(0, 1, 2, 3, 4, 5, 10, 20))
            gen = torch.Generator(device=dev).manual_seed(1)
            pr = roll(model, p0 + (torch.rand(p0.shape, device=dev, generator=gen) * 2 - 1) * 1e-6, v0, 20)
            key = f"d{dens:g}_v{vmax:g}"
            rows = {"balls": nb, "max_speed_t0": float(v0.norm(dim=1).max()), "noise_1e-6": {str(t): diff(pr[t][0], base[t][0]) for t in REP}, "tiled": {}}
            for strips in args.strips:
                for halo in args.halos:
                    one = {}
                    for t0 in (0, 5):
                        p, v = base[t0]
                        hid = torch.zeros(p.shape[0], model.dynamics.hidden_dim, device=dev)
                        with torch.no_grad():
                            gp, gv, _, _ = model.step_free(p, v, hid, render=False)
                            tp, tv, _ = tiled_step(model, p, v, hid, n, strips, halo)
                        one[str(t0)] = {"dp": diff(tp, gp), "dv": diff(tv, gv), "max_speed": float(v.norm(dim=1).max())}
                    r = roll(model, p0, v0, 20, n=n, strips=strips, halo=halo)
                    rows["tiled"][f"S{strips}_h{halo:g}"] = {"one_step": one, "rollout": {str(t): diff(r[t][0], base[t][0]) for t in REP}}
                    print(key, f"S{strips}_h{halo:g}", "one_step dv max t0/t5", one["0"]["dv"]["max"], one["5"]["dv"]["max"],
                          "rollout20", json.dumps(rows["tiled"][f"S{strips}_h{halo:g}"]["rollout"]["20"]), flush=True)
            print(key, "noise20", json.dumps(rows["noise_1e-6"]["20"]), flush=True)
            out[key] = rows
            json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
