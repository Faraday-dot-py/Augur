"""Held-out evaluation of MultiScatterField checkpoints (docs/debugging/scatter-nearfield-retrain.md).

Per case (shape, structure, amplitude 1) and seed set: tick-1 force error (relative L1 of the model acceleration at the initial state, zero hidden
field, vs the fp64 direct sum under the same law), err@5/10/20 (scene-mean of body-mean position error vs fp64 truth), constant-velocity err@20.
Usage: PYTHONPATH=. python scripts/scatter_multi_eval.py --ckpt name=path [--ckpt name2=path2 ...] [--cases all|held] --out results/x.json
       --e-ckpt kern/E_frozen.pt adds the Exp E model on the gravity case.
"""
import argparse
import json

import torch

from scripts import scatter_field as sf
from scripts import scatter_multi as sm

CASES = {
    "grav": ("pow2", "single"), "pow1": ("pow1", "single"), "pow3": ("pow3", "single"), "yuk8": ("yuk8", "single"),
    "yuk128": ("yuk128", "single"), "lj": ("lj", "single"), "charge_pow2": ("pow2", "charge"), "pp_pow2": ("pow2", "predprey"),
    "HELD_pow1.5": ("pow1.5", "single"), "HELD_yuk32": ("yuk32", "single"), "HELD_charge_yuk32": ("yuk32", "charge"),
    "HELD_pp_pow1.5": ("pow1.5", "predprey"),
}
SEEDS = (9000, 12000)


@torch.no_grad()
def eval_case(model, d, dev, is_e=False):
    P, V, mask, typ, cond = d["P"], d["V"], d["mask"], d["typ"], d["cond"]
    S = P.shape[0]
    N = P.shape[2]
    mass = torch.ones(S, N, device=dev)
    if not is_e:
        model.set_law(cond, typ)
    pos, vel = P[:, 0], V[:, 0]
    field = model.init_field(S, dev)
    a, _ = model.force(pos, vel, mass, mask, field[0])
    at = sm.exact_acc(pos.double(), typ, mask.double(), d["params"]).float()
    inside = (pos.abs() < 32).all(-1).float() * mask
    force_rel = float(((a - at).norm(dim=-1) * inside).sum() / (at.norm(dim=-1) * inside).sum())
    errs, p, v = [], pos, vel
    for t in range(1, 21):
        p, v, field, _ = model.step(p, v, mass, mask, field)
        errs.append((((p - P[:, t]).norm(dim=-1) * mask).sum(1) / mask.sum(1)).mean().item())
    cv = (((pos + V[:, 0] * sm.DT * 20 - P[:, 20]).norm(dim=-1) * mask).sum(1) / mask.sum(1)).mean().item()
    return {"force_rel": force_rel, "err5": errs[4], "err10": errs[9], "err20": errs[19], "constvel20": cv, "n": S,
            "finite": bool(torch.isfinite(p).all())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--e-ckpt", default="")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    names = [c for c in CASES if a.cases == "all" or (a.cases == "held" and c.startswith("HELD")) or c in a.cases.split(",")]
    data = {}
    for c in names:
        for s in SEEDS:
            shape, st = CASES[c]
            data[(c, s)] = sm.gen_cases([(shape, st, 1.0)] * a.n, s, dev, 20, tag=2)
    models = {}
    for spec in a.ckpt:
        nm, path = spec.split("=")
        m = sm.build().to(dev).eval()
        st = torch.load(path, map_location=dev, weights_only=False)
        m.load_state_dict(st["model"])
        models[nm] = (m, False, st.get("it"))
    if a.e_ckpt:
        e = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=sm.DT, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
        e.load_state_dict(torch.load(a.e_ckpt, map_location=dev, weights_only=False)["model"])
        models["E"] = (e, True, None)
    res = {}
    for nm, (m, is_e, it) in models.items():
        res[nm] = {"it": it}
        for c in names:
            if is_e and c != "grav":
                continue
            res[nm][c] = {str(s): eval_case(m, data[(c, s)], dev, is_e) for s in SEEDS}
            r = res[nm][c]
            print(nm, c, {s: (round(v["force_rel"], 4), round(v["err20"], 4)) for s, v in r.items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
