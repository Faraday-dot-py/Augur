"""D2b evaluation of a (gated) option-C checkpoint past the training horizon.
 (a) gravity expB scenes (geometry-free): per-tick raw recurrent field max, |a| max, err@20/50/100/150
 (b) contact c_id scenes (several seeds): runaway = max_{t>=21} |a| > 200; per-tick field/|a| growth; err@20/@100 from the scenes' own truth
 (c) scatter_contact_scenes standard families + restitution (cs.run_eval)
  PYTHONPATH=. python scripts/scatter_d2_eval.py --ckpt <pt> --tag <tag> [--seeds 9100,12000] [--fams c_id,...]"""
import argparse
import json
import types

import numpy as np
import torch

from scripts import scatter_contact_scenes as cs
from scripts import scatter_d2b as d2
from scripts import scatter_error_growth as eg
from scripts import scatter_field as sf

KS = (5, 10, 20, 40, 60, 100)


def trace(model, pos, vel, mass, mask, ticks, inj=None):
    field = model.init_field(1, pos.device)
    P, fm, am = [pos[0].cpu().numpy()], [], []
    for t in range(ticks):
        if inj and t == inj[0]:
            field = (field[0] * inj[1],) + tuple(field[1:])
        pos, vel, _, field, _ = sf._step_m(model, pos, vel, mass, mask, field)
        P.append(pos[0].cpu().numpy())
        fm.append(float(field[0].abs().max()))
        am.append(float(field[1].norm(dim=-1).max()))
    return np.stack(P), fm, am


def gravity(model, dev):
    model.set_scene(None, None)
    sc = eg.make_scenes("expB", 8, 9000)
    rows = []
    for s in sc:
        n = len(s["m"])
        P, fm, am = trace(model, s["P"][0].float()[None], s["V"][0].float()[None], s["m"].float()[None], torch.ones(1, n, device=dev), eg.T)
        err = np.linalg.norm(P - s["P"].cpu().numpy(), axis=-1).mean(1)
        rows.append(dict(fm=fm, am=am, err=err))
    at = lambda f, ts: {t: float(f(t)) for t in ts}
    return dict(field_max_median=at(lambda t: np.median([r["fm"][t - 1] for r in rows]), (1, 5, 10, 20, 30, 50, 100, 150)),
                field_max_max=at(lambda t: max(r["fm"][t - 1] for r in rows), (1, 20, 50, 100, 150)),
                a_max_max=at(lambda t: max(max(r["am"][:t]) for r in rows), (20, 50, 100, 150)),
                err=at(lambda t: np.mean([r["err"][t] for r in rows]), (5, 10, 20, 50, 100, 150)))


def contact(model, dev, seed, ticks=100, inj=None):
    es = cs.evalset("c_id", seed, ticks, dev)
    rows = []
    for si, s in enumerate(es["scenes"]):
        n = len(s["mass"])
        pos = torch.tensor(s["pos"], dtype=torch.float32, device=dev)[None]
        vel = torch.tensor(s["vel"], dtype=torch.float32, device=dev)[None]
        model.set_scene(cs._geom_stack([s], dev), cs._rad_pad([s], n).to(dev))
        P, fm, am = trace(model, pos, vel, torch.ones(1, n, device=dev), torch.ones(1, n, device=dev), ticks, inj)
        err = np.linalg.norm(P[:, :n] - es["Pt"][si][: P.shape[0], :n], axis=-1).mean(1)
        off = float((np.abs(np.nan_to_num(P[-1, :n], nan=1e9)).max(-1) > 32).mean())
        rows.append(dict(fm=fm, am=am, err=err, off=off))
    amax = np.array([max(r["am"][21:]) for r in rows])
    return dict(seed=seed, inj=inj, scenes=len(rows), off100=float(np.mean([r['off'] for r in rows])), err_median={t: float(np.median([r['err'][t] for r in rows])) for t in KS}, runaway_frac_gt200=float((amax > 200).mean()), a_max_21plus_median=float(np.median(amax)), a_max_21plus_max=float(amax.max()),
                nonfinite_frac=float(np.mean([not np.isfinite(r["err"][-1]) for r in rows])),
                field_max_median={t: float(np.median([r["fm"][t - 1] for r in rows])) for t in (10, 20, 30, 50, 100)},
                err={t: float(np.mean([r["err"][t] for r in rows])) for t in KS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="9100")
    ap.add_argument("--fams", default="c_id")
    ap.add_argument("--no-families", action="store_true")
    ap.add_argument("--inject", default="", help="tick:lam,... field scaling injected at that tick (large-signal stress)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = d2.build(args.ckpt, dev)
    res = {"ckpt": args.ckpt, "gate": bool(model.gate)}
    with torch.no_grad():
        res["gravity_expB"] = gravity(model, dev)
        print("gravity", json.dumps(res["gravity_expB"]), flush=True)
        res["contact"] = []
        for seed in [int(x) for x in args.seeds.split(",")]:
            r = contact(model, dev, seed)
            res["contact"].append(r)
            print("contact", json.dumps(r), flush=True)
        res["inject"] = []
        for it in [x for x in args.inject.split(",") if x]:
            t, lam = it.split(":")
            for seed in [int(x) for x in args.seeds.split(",")]:
                r = contact(model, dev, seed, inj=(int(t), float(lam)))
                res["inject"].append(r)
                print("inject", json.dumps(r), flush=True)
        if model.gate:
            res["gate_params"] = {k: v.flatten().tolist() for k, v in model.fgate.state_dict().items()}
        if not args.no_families:
            for o in ("kcache", "fastio"):
                sf.apply_opt(model, o)
            cs.run_eval(model, args.tag, types.SimpleNamespace(model="C", ckpt=args.ckpt, fams=args.fams), dev)
    json.dump(res, open(f"results/d2b_eval_{args.tag}.json", "w"), indent=1)
    print("wrote", f"results/d2b_eval_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
