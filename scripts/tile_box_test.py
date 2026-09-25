"""Larger-area test: one trained model, constant density, box/cloud grown beyond the training range.
(a) conservative-contact bouncing balls (density 0.01, speeds +-2.3, gravity on) in boxes n=20..1600 (trained up to n=200),
err@k vs fp64 all-pairs truth, energy/ball vs truth, constant-velocity baseline; (b) translation invariance: same
cluster at two places in a large box, far from walls; (c) CentralForceDynamics N-body at N=1000..8000
(trained 100-1000, scale-init constant density), cutoff 100 and unbounded.

Usage: PYTHONPATH=. python scripts/tile_box_test.py --out results/tile_box_test.json
"""
import argparse
import json
import random

import numpy as np
import torch

import scripts.gravity_sim as gs
from model.central_force import CentralForceDynamics
from model.dataset import make_scenario_uniform
from scripts.ball_1k_rollout import truth as ball_truth
from scripts.energy_fix_common import build_model
from scripts.energy_probe import energy

G, DT = 9.0, 0.15
KS = [1, 5, 10, 20]


def rollout_cons(model, p0, v0, steps):
    pos, vel = p0.clone(), v0.clone()
    hid = torch.zeros(pos.shape[0], model.dynamics.hidden_dim, device=pos.device)
    ps, vs = [pos], [vel]
    with torch.no_grad():
        for _ in range(steps):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            ps.append(pos)
            vs.append(vel)
    return torch.stack(ps), torch.stack(vs)


def make_cons(ckpt, n, nb, device):
    m = build_model("cons", ckpt, n, device)
    m.dynamics.cell_graph = nb > 50
    return m


def boxes(args, dev):
    out = {}
    for n in args.sizes:
        nb = max(2, int(args.density * n * n))
        rows = []
        for seed in args.seeds:
            try:
                balls = make_scenario_uniform(nb, n, 2.3, random.Random(seed))
                tp, tv = ball_truth(balls, n, args.steps, DT, G, 0.75, 400.0, 8, dev)
                tp, tv = torch.from_numpy(tp).to(dev), torch.from_numpy(tv).to(dev)
                model = make_cons(args.checkpoint, n, nb, dev)
                ps, vs = rollout_cons(model, tp[0].clone(), tv[0].clone(), args.steps)
                r = {"seed": seed, "finite": bool(torch.isfinite(ps).all())}
                r["err"] = {str(k): float((ps[k] - tp[k]).norm(dim=1).mean()) for k in KS}
                t = torch.arange(args.steps + 1, device=dev, dtype=torch.float32)[:, None, None] * DT
                cv = tp[0][None] + tv[0][None] * t
                cv[..., 0] += 0.5 * G * t[..., 0] ** 2
                r["constvel_err"] = {str(k): float((cv[k] - tp[k]).norm(dim=1).mean()) for k in KS}
                e, et = energy(ps, vs, G).mean(1), energy(tp, tv, G).mean(1)
                r["E_model_minus_truth"] = {str(k): float(e[k] - et[k]) for k in KS}
                r["E_truth_per_ball"] = float(et[0])
                r["KE_model_truth_20"] = [float(0.5 * (vs[-1] ** 2).sum(-1).mean()), float(0.5 * (tv[-1] ** 2).sum(-1).mean())]
                r["mean_speed_model_truth_20"] = [float(vs[-1].norm(dim=1).mean()), float(tv[-1].norm(dim=1).mean())]
                rows.append(r)
                print("box", n, nb, seed, json.dumps({k: r[k] for k in ("finite", "err", "constvel_err")}), flush=True)
                del tp, tv, ps, vs, model
            except torch.OutOfMemoryError as ex:
                rows.append({"seed": seed, "oom": str(ex)[:80]})
                print("box", n, nb, seed, "OOM", flush=True)
            torch.cuda.empty_cache()
        out[str(n)] = {"balls": nb, "runs": rows}
    return out


def shift_test(args, dev):
    n, nb, steps = 800, 300, 20
    rng = random.Random(4738)
    p0 = torch.tensor([[400.0 + rng.uniform(-50, 50), 400.0 + rng.uniform(-50, 50)] for _ in range(nb)], device=dev)
    v0 = torch.tensor([[rng.uniform(-2.3, 2.3), rng.uniform(-2.3, 2.3)] for _ in range(nb)], device=dev)
    model = make_cons(args.checkpoint, n, nb, dev)
    model.dynamics.cell_graph = True
    base, _ = rollout_cons(model, p0, v0, steps)
    out = {}
    for name, sh in (("aligned_128_0", (128.0, 0.0)), ("offset_137.3_-91.1", (137.3, -91.1)), ("tiny_0.37_0.11", (0.37, 0.11))):
        s = torch.tensor(sh, device=dev)
        moved, _ = rollout_cons(model, p0 + s, v0, steps)
        d = (moved - s - base).norm(dim=-1)
        out[name] = {str(k): {"max": float(d[k].max()), "mean": float(d[k].mean())} for k in (1, 5, 20)}
        print("shift", name, json.dumps(out[name]), flush=True)
    return out


def central(args, dev):
    dyn0 = torch.load(args.central_checkpoint, map_location=dev)
    out = {}
    for N in args.bodies:
        rows = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            p0, v0 = gs.init_bodies(N, rng, scale=True)
            tp, tv = gs.rollout_torch(p0, v0, args.steps, dev)
            r = {"seed": seed, "spread": float(5.0 * (N / 8) ** 0.5)}
            tpt, tvt = torch.from_numpy(tp).to(dev), torch.from_numpy(tv).to(dev)
            r["constvel_err"] = {str(k): float((tpt[0] + tvt[0] * 0.1 * k - tpt[k]).norm(dim=1).mean()) for k in (5, 10, 20)}
            for cutoff in args.cutoffs:
                try:
                    dyn = CentralForceDynamics(dt=0.1, neighbor_radius=cutoff).to(dev)
                    dyn.load_state_dict(dyn0)
                    pos, vel = tpt[0].float().clone(), tvt[0].float().clone()
                    hid = torch.zeros(N, 1, device=dev)
                    ps = [pos]
                    with torch.no_grad():
                        for _ in range(args.steps):
                            dp, dv, hid = dyn(pos, vel, hid)
                            pos, vel = pos + vel * 0.1 + dp, vel + dv
                            ps.append(pos)
                    err = {str(k): float((ps[k].double() - tpt[k]).norm(dim=1).mean()) for k in (5, 10, 20)}
                    e_m = gs_energy(ps[-1].double(), vel.double())
                    e_t = gs_energy(tpt[-1], tvt[-1])
                    r[f"cutoff_{cutoff:g}"] = {"err": err, "E20_model": e_m, "E20_truth": e_t}
                    print("central", N, seed, cutoff, json.dumps(r[f"cutoff_{cutoff:g}"]), flush=True)
                    del dyn, ps
                except torch.OutOfMemoryError as ex:
                    r[f"cutoff_{cutoff:g}"] = {"oom": str(ex)[:80]}
                    print("central", N, seed, cutoff, "OOM", flush=True)
                torch.cuda.empty_cache()
            rows.append(r)
        out[str(N)] = rows
    return out


def gs_energy(pos, vel, eps=0.5):
    d = pos[None] - pos[:, None]
    r = ((d ** 2).sum(-1) + eps ** 2).sqrt()
    iu = torch.triu_indices(len(pos), len(pos), 1, device=pos.device)
    return float(0.5 * (vel ** 2).sum() - (1.0 / r[iu[0], iu[1]]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cons_pure.pt")
    ap.add_argument("--central-checkpoint", default="checkpoints/gravity_central_v1.pt")
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 200, 800, 1600])
    ap.add_argument("--density", type=float, default=0.01)
    ap.add_argument("--bodies", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[100.0, 1e5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[9000, 12000])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/tile_box_test.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = {"args": vars(args)}
    out["shift"] = shift_test(args, dev)
    json.dump(out, open(args.out, "w"))
    out["boxes"] = boxes(args, dev)
    json.dump(out, open(args.out, "w"))
    out["central"] = central(args, dev)
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
