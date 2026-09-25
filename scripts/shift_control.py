"""Translation-invariance test with noise controls (conservative-contact model, 300-ball cluster in an n=800 box, walls out of reach).
Compares shifted-and-unshifted rollouts against (a) run-to-run repeat of the identical input, (b) random position perturbations of
1e-6 and 3e-5 (~fp32 ulp at coordinate 400), then a table of shifts. Also diagnoses why a (0.37, 0.11) shift gave exactly 0.

Usage: PYTHONPATH=. python scripts/shift_control.py --out results/shift_control.json
"""
import argparse
import json
import random

import torch

from scripts.tile_box_test import make_cons, rollout_cons

STEPS = 20
REP = (1, 5, 10, 20)


def stats(a, b):
    d = (a - b).norm(dim=-1)
    return {str(k): {"max": float(d[k].max()), "mean": float(d[k].mean()), "nonzero": int((d[k] > 0).sum())} for k in REP}


def cluster(seed, dev, nb=300):
    rng = random.Random(seed)
    p0 = torch.tensor([[400.0 + rng.uniform(-50, 50), 400.0 + rng.uniform(-50, 50)] for _ in range(nb)], device=dev)
    v0 = torch.tensor([[rng.uniform(-2.3, 2.3), rng.uniform(-2.3, 2.3)] for _ in range(nb)], device=dev)
    return p0, v0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cons_pure.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[4738, 9000, 12000])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", default="results/shift_control.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = {}
    shifts = {"128_0": (128.0, 0.0), "137.3_-91.1": (137.3, -91.1), "0.37_0.11": (0.37, 0.11), "0.5_0": (0.5, 0.0),
              "1_1": (1.0, 1.0), "3.7_1.3": (3.7, 1.3), "-90_60": (-90.0, 60.0)}
    for seed in args.seeds:
        p0, v0 = cluster(seed, dev)
        model = make_cons(args.checkpoint, 800, p0.shape[0], dev)
        model.dynamics.cell_graph = True
        base, _ = rollout_cons(model, p0, v0, STEPS)
        r = {"repeat_identical": stats(rollout_cons(model, p0, v0, STEPS)[0], base)}
        gen = torch.Generator(device=dev).manual_seed(seed)
        for name, mag in (("perturb_1e-6", 1e-6), ("perturb_3e-5", 3e-5)):
            rs = []
            for _ in range(args.draws):
                noise = (torch.rand(p0.shape, device=dev, generator=gen) * 2 - 1) * mag
                rs.append(stats(rollout_cons(model, p0 + noise, v0, STEPS)[0], base))
            r[name] = rs
        r["shifts"] = {}
        for name, sh in shifts.items():
            s = torch.tensor(sh, device=dev)
            moved, _ = rollout_cons(model, p0 + s, v0, STEPS)
            st = stats(moved - s, base)
            st["roundtrip_exact_frac"] = float((((p0 + s) - s) == p0).all(dim=1).float().mean())
            st["moved_minus_base_equals_shift_frac_step20"] = float(((moved[20] - base[20]) == s).all(dim=1).float().mean())
            r["shifts"][name] = st
            print(seed, name, json.dumps({k: st[k] for k in ("1", "20", "roundtrip_exact_frac")}), flush=True)
        out[str(seed)] = r
        print(seed, "repeat", json.dumps(r["repeat_identical"]), flush=True)
        print(seed, "perturb_3e-5", json.dumps([x["20"] for x in r["perturb_3e-5"]]), flush=True)
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
