"""Evaluate generalized-systems checkpoints on the original solar system and on training-family and held-out-family planetary systems, vs the solar-scale-same-horizon-control and orbit_ft2, identical seeds and scenes (edge fix on).

  python scripts/eval_scatter_solar_generalized.py --baseline orbit_ft2.pt --arm solar-scale-same-horizon-control=ckpt.pt --arm solar-scale-generalized-systems-control-recipe=ckpt.pt --pick solar-scale-generalized-systems-control-recipe

Seeds: 4738 first, 9100/9200 selection, 9300/12000 confirmation (verdict judged there only). Per family and seed: --per-seed scenes from scatter_solar_generalized.eval_rng(seed, family, j); the exact integrator must keep them stable over all --steps ticks (eval scenes only; training used 500 ticks).
Pre-registered verdict (per arm): PASS = every held-out family keeps >= 75% of planets at tick 2000 on the confirmation seeds AND the original solar system keeps mean >= 6 of 8 on 9300/12000 AND the held-out mean beats the same-horizon control;
KILL = mean over the four held-out families < 50%; else INCONCLUSIVE.
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import scatter_scenes as ss
from scripts import scatter_solar as sol
from scripts import scatter_solar_generalized as gen
from scripts import train_scatter_solar_stability as tr

TICKS = [100, 500, 1000, 2000]
ORIGINAL = "original-solar"


def solar_as_general(scene):
    kind = scene["kind"]
    host, hl = np.full(len(kind), -1), np.zeros(len(kind))
    for j in np.where(kind == 2)[0]:
        host[j] = sol.MOON_HOST
        hl[j] = gen.hill(np.linalg.norm(scene["pos"][sol.MOON_HOST] - scene["pos"][0]), scene["mass"][sol.MOON_HOST], sol.STAR_M)
    return dict(scene, host=host, hill=hl)


def frac_kept(met, tick=2000):
    return met["planets"][str(tick)]["kept"], met["n_planets"]


def pooled(table, model, family, seeds, tick):
    k = n = 0.0
    for s in seeds:
        for m in table[family][str(s)][model]:
            a, b = m["planets"][str(tick)]["kept"], m["n_planets"]
            k, n = k + a, n + b
    return k / n


def verdict(table, arm, control, confirm, families=gen.HELD_OUT_FAMILIES):
    fam = {f: pooled(table, arm, f, confirm, 2000) for f in families}
    ctl = {f: pooled(table, control, f, confirm, 2000) for f in families}
    orig = float(np.mean([sum(m["planets"]["2000"]["kept"] for m in table[ORIGINAL][str(s)][arm]) for s in confirm]))
    mean_arm, mean_ctl = float(np.mean(list(fam.values()))), float(np.mean(list(ctl.values())))
    if mean_arm < 0.5:
        res = "KILL"
    elif min(fam.values()) >= 0.75 and orig >= 6.0 and mean_arm > mean_ctl:
        res = "PASS"
    else:
        res = "INCONCLUSIVE"
    return {"result": res, "held_out_kept_fraction_2000_confirm": fam, "control_held_out_kept_fraction_2000_confirm": ctl, "held_out_mean": mean_arm, "control_held_out_mean": mean_ctl, "original_solar_mean_kept_2000_confirm": orig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--baseline-name", default="orbit_ft2")
    ap.add_argument("--arm", action="append", default=[], help="name=ckpt (include solar-scale-same-horizon-control)")
    ap.add_argument("--control-name", default="solar-scale-same-horizon-control")
    ap.add_argument("--pick", action="append", default=[], help="arm names to give a verdict")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--per-seed", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=4738)
    ap.add_argument("--select-seeds", default="9100,9200")
    ap.add_argument("--confirm-seeds", default="9300,12000")
    ap.add_argument("--kpad", choices=["zero", "ext"], default="ext")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/solar_generalized")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    sel = [int(x) for x in args.select_seeds.split(",")]
    conf = [int(x) for x in args.confirm_seeds.split(",")]
    seeds = [args.first_seed] + sel + conf
    ticks = [t for t in TICKS if t <= args.steps]
    ckpts = {args.baseline_name: args.baseline, **dict(a.split("=", 1) for a in args.arm)}
    models = {n: tr.build_model(c, dev, args.kpad).eval() for n, c in ckpts.items()}
    table = {}
    stats = {f: {} for f in gen.EVAL_FAMILIES}
    keep = {}
    for family in [ORIGINAL] + list(gen.EVAL_FAMILIES):
        table[family] = {}
        for seed in seeds:
            if family == ORIGINAL:
                scenes = [solar_as_general(sol.solar_scene(seed, 200, tr.HILL_K))]
                truth = [ss.nbody_truth(s["pos"], s["vel"], s["mass"], args.steps, dev) for s in scenes]
            else:
                scenes, truth = [], []
                for j in range(args.per_seed):
                    sc, Pt, Vt = gen.make_scene(gen.eval_rng(seed, family, j), family, args.steps, args.steps, dev, stats[family])
                    scenes.append(sc)
                    truth.append((Pt, Vt))
            table[family][str(seed)] = {}
            for name, model in models.items():
                Pm, Vm, _ = ss.model_rollout(model, scenes, args.steps, dev)
                mets = []
                for i, sc in enumerate(scenes):
                    n = len(sc["mass"])
                    mets.append(gen.family_metrics(Pm[i, :, :n], Vm[i, :, :n], truth[i][0], truth[i][1], sc, ticks))
                    if seed == args.first_seed and i == 0:
                        keep[f"{family}__model_{name}"] = Pm[i, :, :n].astype(np.float32)
                        keep[f"{family}__truth"], keep[f"{family}__mass"], keep[f"{family}__kind"] = truth[i][0].astype(np.float32), sc["mass"], sc["kind"]
                table[family][str(seed)][name] = mets
                print(family, seed, name, "kept", {str(t): round(sum(m["planets"][str(t)]["kept"] for m in mets), 2) for t in ticks}, "of", sum(m["n_planets"] for m in mets), flush=True)
    for f in stats:
        st = stats[f]
        st["rejection_rate"] = 1.0 - st.get("accepted", 0) / max(1, st.get("attempts", 1))
    out = {"seeds": {"first": args.first_seed, "select": sel, "confirm": conf}, "ckpts": ckpts, "per_seed": args.per_seed, "eval_scene_stability": stats, "table": table}
    out["verdicts"] = {a: verdict(table, a, args.control_name, conf) for a in args.pick}
    for a, v in out["verdicts"].items():
        print("VERDICT", a, json.dumps(v), flush=True)
    np.savez_compressed(f"{args.out_dir}/traj_seed{args.first_seed}.npz", **keep)
    json.dump(out, open(f"{args.out_dir}/eval_summary.json", "w"), indent=1)
    print("DONE eval", flush=True)


if __name__ == "__main__":
    main()
