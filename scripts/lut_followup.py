"""Follow-up diagnostics for the LUT / adaptive results (GPU):
(1) interior table error (excluding the last segment) for pair/wall tables, and wall-table range hi in {2, 8, 32}
(2) mixed_t100 state: max wall penetration, ball with the largest LUT one-step diff, and one-step diff with hi=8/32
(3) sparse_t0 skin1: ball with the largest one-step diff, its neighbours and wall distances, active flag
Usage: PYTHONPATH=. python scripts/lut_followup.py --out results/lut_followup.json
"""
import argparse
import json
import random

import torch

import scripts.lut_adaptive as la
from model.dataset import make_scenario_uniform
from scripts.energy_fix_common import build_model
from scripts.lut_eval import advance

DEV = "cuda"


def install_lut(model, lut, cubic, wall_hi):
    dyn = model.dynamics
    dyn.__dict__.pop("forward", None)
    fc = la.FastContact(dyn, lut, cubic, None)
    pair = lambda x: x * dyn.pair_force(x[:, None])[:, 0] * dyn.force_scale
    wall = lambda x: x * dyn.wall_force(x[:, None])[:, 0] * dyn.force_scale
    fc.wall_t = la.Table(wall, 0.0, wall_hi, lut, cubic).to(dyn.gravity.device)
    fc.wall_g = fc.wall_t
    dyn.forward = fc
    return fc


def step(model, p, v):
    hid = torch.zeros(p.shape[0], 1, device=p.device)
    with torch.no_grad():
        a, b, _, _ = model.step_free(p, v, hid, render=False)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cons_pure.pt")
    ap.add_argument("--out", default="results/lut_followup.json")
    args = ap.parse_args()
    dev = torch.device(DEV)
    n = 317
    model = build_model("cons", args.checkpoint, n, dev)
    model.dynamics.cell_graph = False
    dyn = model.dynamics
    out = {}
    with torch.no_grad():
        interior = {}
        for lut in (256, 1024, 4096):
            for cubic in (False, True):
                fc = la.FastContact(dyn, lut, cubic, None)
                r = {}
                for name, hi, fn, tab in (("pair", 1.0, lambda x: x * dyn.pair_force(x[:, None])[:, 0] * dyn.force_scale, fc.pair_t),
                                          ("wall", 2.0, lambda x: x * dyn.wall_force(x[:, None])[:, 0] * dyn.force_scale, fc.wall_t)):
                    x = torch.rand(400000, device=dev) * hi * (1 - 2.0 / lut)
                    e = (fn(x) - tab(x)).abs()
                    r[name] = {"max_abs_interior": float(e.max()), "rms": float(e.pow(2).mean().sqrt())}
                interior[f"lut{lut}{'c' if cubic else ''}"] = r
        out["interior_error"] = interior
        xs = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0], device=dev)
        out["wall_mlp_g"] = {str(float(x)): float(x * dyn.wall_force(x.view(1, 1))[0, 0] * dyn.force_scale) for x in xs}
    print(json.dumps(out), flush=True)
    balls = make_scenario_uniform(1000, n, 2.3, random.Random(4738))
    p0 = torch.tensor([[b["x"], b["y"]] for b in balls], device=dev)
    v0 = torch.tensor([[b["vx"], b["vy"]] for b in balls], device=dev)
    p, v = advance(model, p0, v0, 100)
    x, y = p[:, 0], p[:, 1]
    wd = torch.stack([x, (n - 1) - x, y, (n - 1) - y], 1)
    out["mixed_t100"] = {"max_wall_pen": float(((0.75 - wd).clamp(min=0) / 0.75).max()), "n_pen_gt2": int((((0.75 - wd).clamp(min=0) / 0.75) > 2).any(1).sum()),
                         "max_speed": float(v.norm(dim=1).max())}
    la.install(model, "base")
    bp, bv = step(model, p, v)
    for hi in (2.0, 8.0, 32.0):
        install_lut(model, 1024, False, hi)
        sp, sv = step(model, p, v)
        d = (sv - bv).norm(dim=1)
        k = int(d.argmax())
        out["mixed_t100"][f"lut1024_wallhi{hi}"] = {"max_dv": float(d.max()), "argmax_ball": k, "argmax_wall_pen": float(((0.75 - wd[k]).clamp(min=0) / 0.75).max()),
                                                    "argmax_speed": float(v[k].norm())}
    la.install(model, "base")
    print(json.dumps(out["mixed_t100"]), flush=True)
    la.install(model, "base")
    bp, bv = step(model, p0, v0)
    fc = la.install(model, "skin1")
    sp, sv = step(model, p0, v0)
    d = (sv - bv).norm(dim=1)
    top = d.topk(3).indices.tolist()
    act = fc.active_mask(p0, v0)
    rows = []
    for k in top:
        dist = (p0 - p0[k]).norm(dim=1)
        dist[k] = 1e9
        nb = dist.topk(3, largest=False)
        x0, y0 = p0[k]
        rows.append({"ball": k, "dv": float(d[k]), "active": bool(act[k]), "pos": [float(x0), float(y0)], "vel": [float(a) for a in v0[k]],
                     "wall_dist": [float(x0), float(n - 1 - x0), float(y0), float(n - 1 - y0)],
                     "nbr_dist": [float(a) for a in nb.values], "nbr_active": [bool(act[i]) for i in nb.indices], "nbr_speed": [float(v0[i].norm()) for i in nb.indices],
                     "dv_vec_base": [float(a) for a in (bv[k] - v0[k])], "dv_vec_skin1": [float(a) for a in (sv[k] - v0[k])]})
    out["sparse_t0_skin1"] = rows
    print(json.dumps(rows), flush=True)
    json.dump(out, open(args.out, "w"))
    print("done")


if __name__ == "__main__":
    main()
