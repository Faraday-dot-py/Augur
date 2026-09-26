"""genic tasks 1+2: static accuracy/cost of the flyby-trained estimator, geometric dual tree and BH on unseen ICs (N=100k, steps 0/300/1000
of exact rollouts), calibration on freshly labelled pairs, and the retrained-estimator generalization gap.

Usage: PYTHONPATH=. python scripts/genic_static.py --ics three,plummer --out results/genic_static_A.json
"""
import argparse
import json

import torch

from scripts import adaptive_oracle as ao
from scripts import est_train
from scripts import genic_lib as gl
from scripts import kernels
from scripts import nbody_ic

STEPS = (0, 300, 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ics", default="three,plummer,disk,clumpy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--extra", action="store_true", help="also uniform and flyby_t0/t1000/t5000 reference states")
    a = ap.parse_args()
    dev = torch.device("cuda")
    k = kernels.analytic()
    kernels.current = k
    base = est_train.load("checkpoints/est_analytic.pt", dev)
    gen = torch.Generator(device=dev).manual_seed(4738)
    ics = a.ics.split(",")

    def state(ic, step):
        if step == 0:
            return nbody_ic.IC[ic](100000, k, dev)[0]
        return gl.load_snap_state(f"genic_ex100k_{ic}", step, dev)

    out = {"static": {}, "calib": {}}
    if a.extra:
        args = type("A", (), {"npz": "results/flyby_100k.npz", "bodies": 100000})()
        for s in ("uniform", "flyby_t0", "flyby_t1000", "flyby_t5000"):
            pos = ao.load_state(s, args, dev)
            S = gl.make_targets(len(pos), 20000, dev)
            out["static"][s] = gl.sweep(pos, {"flyby": base}, S)
            out["calib"][s] = gl.calib(pos, base, gen)
            print(s, json.dumps(out["calib"][s]), flush=True)
    if not a.retrain:
        for ic in ics:
            for st in STEPS:
                key = f"{ic}_{st}"
                pos = state(ic, st)
                S = gl.make_targets(len(pos), 20000, dev)
                out["static"][key] = gl.sweep(pos, {"flyby": base}, S)
                out["calib"][key] = gl.calib(pos, base, gen)
                print(key, json.dumps(out["calib"][key]), flush=True)
                for r in out["static"][key]["est"]["flyby"]:
                    print(key, "est", json.dumps(r), flush=True)
                json.dump(out, open(a.out, "w"))
    else:
        data = {}
        for ic in ics:
            for st in STEPS:
                data[(ic, st)] = gl.pair_data(state(ic, st), gen)
                print("labelled", ic, st, len(data[(ic, st)][1]), flush=True)
        out["retrain"] = {}
        heads = {"all_ics": gl.train_on([data[(ic, st)] for ic in ics for st in (0, 300)], dev)}
        for ic in ics:
            heads[f"loo_{ic}"] = gl.train_on([data[(j, st)] for j in ics if j != ic for st in (0, 300)], dev)
        for ic in ics:
            key = f"{ic}_1000"
            pos = state(ic, 1000)
            S = gl.make_targets(len(pos), 20000, dev)
            hs = {"flyby": base, "all_ics": heads["all_ics"], "loo": heads[f"loo_{ic}"]}
            out["retrain"][key] = gl.sweep(pos, hs, S, geo=True, bh=False)
            out["retrain"][key]["calib"] = {n: gl.calib(pos, h, gen) for n, h in hs.items()}
            print(key, json.dumps(out["retrain"][key]["calib"]), flush=True)
            json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
