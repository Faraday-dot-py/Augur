"""Oracle bound and held-out scoring for the residual corrector on the frozen matched tier 0 (tiled hierarchy, arm OR).
  oracle: frozen arm OR with the force partly replaced by the exact force at the model's own state (assimilation-style); gap closure
          = (frozen - oracle) / (frozen - BL) on err@20 and dE@100, BL = tiled Exp E + far kernel from results/loss_term_arms/baseline_hier_eval_matched60k.json.
  eval:   frozen OR vs each corrector checkpoint on seeds 9100/9200/9300 (24 scenes x 100 ticks) plus plain untiled scenes (single window,
          the train_tier0_ext plain eval scenes) with and without the head.
Usage: PYTHONPATH=. python scripts/eval_tier0_corrector.py --mode oracle --s1 <ckpt> --out results/tier0_corrector/oracle.json"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf
from scripts import train_tiered as ttr
from scripts.tier0_corrector import Oracle, PlainBase, ResidualCorrector, TiledBase, make_acc, score

SEEDS = (9100, 9200, 9300)
ORACLES = [("lam0.25", "lam", 0.25), ("lam0.5", "lam", 0.5), ("lam0.75", "lam", 0.75), ("steps2", "steps", 2), ("steps5", "steps", 5),
           ("steps10", "steps", 10), ("parts0.1", "parts", 0.1), ("parts0.5", "parts", 0.5), ("dense0.1", "dense", 0.1), ("lam1", "lam", 1.0)]


def load_corr(path, dev):
    st = torch.load(path, map_location=dev, weights_only=False)
    a = st["args"]
    c = ResidualCorrector(width=a["width"], use_ext=a["use_ext"], momfix=not a["no_momfix"]).to(dev)
    c.load_state_dict(st["corr"])
    return c.eval().requires_grad_(False), st


def bl_numbers(path):
    d = json.load(open(path))["BL"]
    return {s: {"err@20": d[s]["err@20"], "dE@100": d[s]["dE_model"]} for s in d}


def closure(frozen, oracle, bl):
    out = {}
    for key in ("err@20", "dE@100"):
        gap = np.mean([frozen[s][key] - bl[s][key] for s in frozen])
        win = np.mean([frozen[s][key] - oracle[s][key] for s in frozen])
        out[key] = float(win / gap) if abs(gap) > 1e-12 else None
    return out


def run_oracle(args, dev, log):
    base = TiledBase(args.s1, dev, chunk=args.chunk)
    bl = bl_numbers(args.bl)
    res = {"frozen": {}}
    data = {sd: ttr.make_scenes(args.eval_scenes, args.eval_steps, sd, dev) for sd in SEEDS}
    for sd in SEEDS:
        res["frozen"][str(sd)] = score(make_acc(base), base, data[sd])
        log(f"frozen {sd} {json.dumps(res['frozen'][str(sd)])}")
    for name, kind, val in ORACLES:
        res[name] = {}
        for sd in SEEDS:
            res[name][str(sd)] = score(make_acc(base, None, Oracle(kind, val)), base, data[sd])
            log(f"{name} {sd} {json.dumps(res[name][str(sd)])}")
        res[name]["closure"] = closure(res["frozen"], res[name], bl)
        log(f"{name} closure {res[name]['closure']}")
        json.dump(res, open(args.out, "w"), indent=1)
    res["bl"] = bl


def plain_score(model, corr, dev, seeds, args):
    """err@20 and dE@100 on the plain eval scenes with the head patched into model.force (tick = force-call index since init_field)."""
    out = {}
    orig_force, orig_init = model.force, model.init_field
    state = {"c": 0}
    if corr is not None:
        base = PlainBase(model)

        def force(pos, vel, mass, mask, field):
            a, f = orig_force(pos, vel, mass, mask, field)
            a = a + corr(pos, a, mask, base.edge(pos), state["c"], None)
            state["c"] += 1
            return a, f

        def init_field(*a, **k):
            state["c"] = 0
            return orig_init(*a, **k)

        model.force, model.init_field = force, init_field
    try:
        model.t0.ext_grid = model.t0.ext_poly = None
        model.t0.set_material(None)
        for sd in seeds:
            data = tsf.make_data(tsf.EXPS["E"], 48, 100, sd, dev, 0.1, 0.5)
            with torch.no_grad():
                m, _ = tsf.evaluate_model(model, data, dev, 0.1, 0.5)
            out[str(sd)] = {"err@20": m["err"][19], "err@100": m["err"][99], "dE@100": m["energy_drift_rel_model"][99]}
    finally:
        if corr is not None:
            del model.force, model.init_field
    return out


def run_eval(args, dev, log):
    from scripts import tiered_adapters as ta

    base = TiledBase(args.s1, dev, chunk=args.chunk)
    bl = bl_numbers(args.bl)
    corrs = {os.path.basename(p).replace(".pt", ""): load_corr(p, dev)[0] for p in args.corr}
    res = {"bl": bl, "tiled": {"frozen": {}}, "plain": {}}
    model = ta.load_t0ext(args.s1, dev).m
    arms = {"frozen": None, **corrs}
    for name, c in arms.items():
        res["plain"][name] = plain_score(model, c, dev, tsf.EVAL_SEEDS, args)
        log(f"plain {name} {json.dumps(res['plain'][name])}")
    for sd in SEEDS:
        d = ttr.make_scenes(args.eval_scenes, args.eval_steps, sd, dev)
        for name, c in arms.items():
            res["tiled"].setdefault(name, {})[str(sd)] = score(make_acc(base, c), base, d)
            log(f"tiled {name} {sd} {json.dumps(res['tiled'][name][str(sd)])}")
        json.dump(res, open(args.out, "w"), indent=1)
    verdicts = {}
    plain_f = np.mean([res["plain"]["frozen"][s]["err@20"] for s in res["plain"]["frozen"]])
    for name in corrs:
        e = {s: res["tiled"][name][s] for s in map(str, SEEDS)}
        verdicts[name] = {
            "err@20_over_BL": {s: e[s]["err@20"] / bl[s]["err@20"] for s in e},
            "dE@100_over_BL": {s: e[s]["dE@100"] / bl[s]["dE@100"] for s in e},
            "err@20_over_frozen": {s: e[s]["err@20"] / res["tiled"]["frozen"][s]["err@20"] for s in e},
            "plain_err@20_over_frozen": float(np.mean([res["plain"][name][s]["err@20"] for s in res["plain"][name]]) / plain_f)}
        v = verdicts[name]
        v["PASS"] = bool(all(x <= 1.25 for x in v["err@20_over_BL"].values()) and all(x <= 2.5 for x in v["dE@100_over_BL"].values())
                         and v["plain_err@20_over_frozen"] <= 1.1)
    res["verdicts"] = verdicts
    json.dump(res, open(args.out, "w"), indent=1)
    log(json.dumps(verdicts, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "eval"], required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--corr", nargs="*", default=[])
    ap.add_argument("--bl", default="results/loss_term_arms/baseline_hier_eval_matched60k.json")
    ap.add_argument("--eval-scenes", type=int, default=24)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log = lambda s: print(s, flush=True)
    {"oracle": run_oracle, "eval": run_eval}[args.mode](args, dev, log)


if __name__ == "__main__":
    main()
