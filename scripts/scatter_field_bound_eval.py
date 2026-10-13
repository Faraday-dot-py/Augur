"""D1: soft bound on the recurrent field fed back into the grid net, inference only, zero-shot on the option-C fine-tune and on Exp E.
See docs/debugging/error-correction-D1-bounded-field.md.

  PYTHONPATH=. python scripts/scatter_field_bound_eval.py --ckpt-c checkpoints_contact/C.pt --ckpt-e <E.pt> --out results/errgrowth/d1_bounded_field.json
The bound is a forward-pre-hook on model.net that maps the fed-back field input channels x[:, 5:5+cf] through f(.); nothing else changes.
"""
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts import scatter_contact as sct
from scripts import scatter_contact_scenes as cs
from scripts import scatter_error_growth as eg
from scripts import scatter_error_growth_contact2 as c2
from scripts import scatter_field_bound_cons as bc
from scripts import scatter_scenes as ss
from scripts import train_scatter_field as tsf

FIELD0 = 5
VARIANTS = (("base", None, 0.0), ("tanh100", "tanh", 100.0), ("tanh200", "tanh", 200.0), ("tanh400", "tanh", 400.0),
            ("alg100", "alg", 100.0), ("alg200", "alg", 200.0))
STATE = {"form": None, "s": 0.0}


def squash(f, form, s):
    if form == "tanh":
        return s * torch.tanh(f / s)
    return f / torch.sqrt(1 + (f / s) ** 2)


def install(model):
    cf = model.cf

    def pre(mod, inp):
        if STATE["form"] is None:
            return None
        x = inp[0].clone()
        x[:, FIELD0:FIELD0 + cf] = squash(x[:, FIELD0:FIELD0 + cf], STATE["form"], STATE["s"])
        return (x,) + tuple(inp[1:])

    model.net.register_forward_pre_hook(pre)


def setv(v):
    STATE["form"], STATE["s"] = v[1], v[2]


def gravity_rows(net, scenes, steps):
    rows = []
    for s in scenes:
        rec = []
        Pm, _ = eg.roll(net, s, steps=steps, rec=rec)
        ep = (Pm - s["P"][: steps + 1]).norm(dim=-1).mean(1).cpu().numpy()
        rows.append(dict(ep=ep, a=max(r["a_max"] for r in rec[20:]), f=max(r["fld_max"] for r in rec)))
    return dict(err={k: float(np.mean([r["ep"][k] for r in rows])) for k in (5, 20, 50, 100, steps)},
                runaway=float(np.mean([r["a"] > 200 for r in rows])), a_max_median=float(np.median([r["a"] for r in rows])),
                fld_max_median=float(np.median([r["f"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-c", required=True)
    ap.add_argument("--ckpt-e", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-cons", action="store_true")
    ap.add_argument("--ticks", type=int, default=100)
    args = ap.parse_args()
    dev = eg.DEV
    res = {}
    mc = cs.build("C", dev, args.ckpt_c)
    me = cs.build("E", dev, args.ckpt_e)
    install(mc)
    install(me)
    nets = {"C": c2.NetC(mc), "E": eg.Net(me)}
    models = {"C": mc, "E": me}
    t0 = time.time()
    scenes = {"expB": eg.make_scenes("expB", 12, 9000), "cluster": eg.make_scenes("cluster", 6, 9100)}
    print("scenes built", round(time.time() - t0), flush=True)
    rn = SimpleNamespace(variant="ms_kp_pot_v_g128", exp="E", dt=0.1, eps=0.5, opt="none")
    plain = {}
    for name, ck in (("C", args.ckpt_c), ("E", args.ckpt_e)):
        plain[name] = tsf.build(tsf.EXPS["E"], rn.variant, rn.dt).to(dev)
        plain[name].load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"], strict=False)
        plain[name].eval()
        install(plain[name])
    evals = {fam: [cs.evalset(fam, sd, args.ticks, dev) for sd in ss.EVAL_SEEDS] for fam in ("c_id",)}
    for v in VARIANTS:
        setv(v)
        r = {}
        with torch.no_grad():
            mc.net.geo = None
            mc.momfix = True
            for name in ("C", "E"):
                for fam, sc in scenes.items():
                    r[f"{name}_grav_{fam}"] = gravity_rows(nets[name], sc, 150)
            print(v[0], "gravity", json.dumps({k: (x["err"][20], x["err"][100], x["runaway"]) for k, x in r.items()}), flush=True)
            for name in ("C", "E"):
                fr, _ = cs.eval_family(models[name], "c_id", dev, args.ticks)
                r[f"{name}_contact_c_id"] = fr["mean"]
            args_c2 = SimpleNamespace(scenes=24, ticks=args.ticks)
            runs = [c2.contact_run(mc, es, args_c2, dev) for es in evals["c_id"]]
            rows = [x for rr in runs for x in rr]
            tm = np.array([max(x["tot_max"][21:]) for x in rows])
            r["C_contact_runaway"] = dict(frac_gt_200=float((tm > 200).mean()), median_tot_max=float(np.median(tm)), n=len(rows),
                                          err_at_5_10_20_40_end=[float(np.mean([x["err"][i] for x in rows])) for i in range(5)],
                                          fld_max_t10_t30_t55_median=[float(np.median([x["fld_max"][t] for x in rows])) for t in (10, 30, 55)])
            print(v[0], "contact runaway", json.dumps(r["C_contact_runaway"]), flush=True)
            for name in ("C", "E"):
                bx = [sct.hbox((-7, -7), (7, 7))]
                r[f"{name}_rest_wall"] = cs.restitution(models[name], dev, bx, 7.0, (2, 4, 6, 8, 10, 12, 15))
                r[f"{name}_rest_circle"] = cs.restitution(models[name], dev, bx + [sct.circle((2.0, 0.0), 2.0)], 0.0, (2, 4, 6, 8, 10))
            print(v[0], "restitution", json.dumps({k: x for k, x in r.items() if "rest" in k}), flush=True)
            if not args.no_cons:
                for name in ("C", "E"):
                    r[f"{name}_cons"] = bc.conservation_rows(plain[name], dev, rn)
        res[v[0]] = r
        json.dump(res, open(args.out, "w"))
        print(v[0], "done", round(time.time() - t0), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
