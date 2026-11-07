"""Evaluate solar-scale-translation-augmentation-retrain checkpoints on the same scenes and seeds as eval_scatter_solar_generalized.py (original solar system + 5 families, seeds 4738, 9100, 9200, 9300, 12000, 2 scenes per family and seed, 2000 ticks, edge fix on), optionally with the whole scene (and its truth) translated by --offset.

  python scripts/eval_scatter_solar_translation.py --baseline orbit_ft2.pt --arm NAME=ckpt.pt ... --pick solar-scale-translation-augmentation-control-recipe --control solar-scale-generalized-systems-control-recipe

Extra metric over the generalized eval: moon survival time (first tick with moon-host separation above the host Hill radius, tick 2000 if never) for model and truth.
Pre-registered verdict (per picked arm): PASS = every held-out family keeps >= 75% of its planets at tick 2000 (pooled, confirmation seeds 9300/12000) AND the original solar system keeps >= 6 of 8 on seed 9300 and on seed 12000 AND the held-out mean kept fraction beats --control's;
KILL = held-out mean < 50% or below --control's; otherwise INCONCLUSIVE.
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
from scripts.eval_scatter_solar_generalized import ORIGINAL, TICKS, pooled, solar_as_general


def moon_survival(P, scene):
    out = []
    for j in np.where(scene["kind"] == 2)[0]:
        sep = np.linalg.norm(P[:, j] - P[:, scene["host"][j]], axis=-1)
        out.append(int(np.argmax(sep > scene["hill"][j])) if (sep > scene["hill"][j]).any() else int(P.shape[0] - 1))
    return out


def verdict(table, arm, control, confirm, original_seeds):
    fam = {f: pooled(table, arm, f, confirm, 2000) for f in gen.HELD_OUT_FAMILIES}
    ctl = {f: pooled(table, control, f, confirm, 2000) for f in gen.HELD_OUT_FAMILIES}
    orig = {str(s): float(sum(m["planets"]["2000"]["kept"] for m in table[ORIGINAL][str(s)][arm])) for s in original_seeds}
    mean_arm, mean_ctl = float(np.mean(list(fam.values()))), float(np.mean(list(ctl.values())))
    if mean_arm < 0.5 or mean_arm < mean_ctl:
        res = "KILL"
    elif min(fam.values()) >= 0.75 and min(orig.values()) >= 6.0 and mean_arm > mean_ctl:
        res = "PASS"
    else:
        res = "INCONCLUSIVE"
    return {"result": res, "held_out_kept_fraction_2000_confirm": fam, "control_held_out_kept_fraction_2000_confirm": ctl, "held_out_mean": mean_arm, "control_held_out_mean": mean_ctl, "original_solar_kept_2000_confirm": orig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--baseline-name", default="orbit_ft2")
    ap.add_argument("--arm", action="append", default=[], help="name=ckpt")
    ap.add_argument("--control", default="solar-scale-generalized-systems-control-recipe")
    ap.add_argument("--pick", action="append", default=[])
    ap.add_argument("--offset", default="0,0", help="whole-scene translation x,y applied to the initial state and the truth")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--per-seed", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=4738)
    ap.add_argument("--select-seeds", default="9100,9200")
    ap.add_argument("--confirm-seeds", default="9300,12000")
    ap.add_argument("--kpad", choices=["zero", "ext"], default="ext")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/solar_translation")
    ap.add_argument("--out-name", default="eval_summary")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    off = np.array([float(x) for x in args.offset.split(",")])
    sel = [int(x) for x in args.select_seeds.split(",")]
    conf = [int(x) for x in args.confirm_seeds.split(",")]
    seeds = [args.first_seed] + sel + conf
    ticks = [t for t in TICKS if t <= args.steps]
    ckpts = {args.baseline_name: args.baseline, **dict(a.split("=", 1) for a in args.arm)}
    models = {n: tr.build_model(c, dev, args.kpad).eval() for n, c in ckpts.items()}
    table, keep = {}, {}
    stats = {f: {} for f in gen.EVAL_FAMILIES}
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
            truth = [(Pt + off, Vt) for Pt, Vt in truth]
            scenes = [dict(sc, pos=sc["pos"] + off) for sc in scenes]
            table[family][str(seed)] = {}
            for name, model in models.items():
                Pm, Vm, _ = ss.model_rollout(model, scenes, args.steps, dev)
                mets = []
                for i, sc in enumerate(scenes):
                    n = len(sc["mass"])
                    met = gen.family_metrics(Pm[i, :, :n], Vm[i, :, :n], truth[i][0], truth[i][1], sc, ticks)
                    met["moon_survival_model"], met["moon_survival_truth"] = moon_survival(Pm[i, :, :n], sc), moon_survival(truth[i][0], sc)
                    mets.append(met)
                    if seed == args.first_seed and i == 0:
                        keep[f"{family}__model_{name}"] = Pm[i, :, :n].astype(np.float32)
                        keep[f"{family}__truth"], keep[f"{family}__mass"], keep[f"{family}__kind"] = truth[i][0].astype(np.float32), sc["mass"], sc["kind"]
                table[family][str(seed)][name] = mets
                print(family, seed, name, "kept", {str(t): round(sum(m["planets"][str(t)]["kept"] for m in mets), 2) for t in ticks}, "of", sum(m["n_planets"] for m in mets), flush=True)
    for f in stats:
        st = stats[f]
        st["rejection_rate"] = 1.0 - st.get("accepted", 0) / max(1, st.get("attempts", 1))
    out = {"offset": off.tolist(), "seeds": {"first": args.first_seed, "select": sel, "confirm": conf}, "ckpts": ckpts, "per_seed": args.per_seed, "eval_scene_stability": stats, "table": table}
    out["verdicts"] = {a: verdict(table, a, args.control, conf, conf) for a in args.pick}
    for a, v in out["verdicts"].items():
        print("VERDICT", a, json.dumps(v), flush=True)
    np.savez_compressed(f"{args.out_dir}/{args.out_name}_traj_seed{args.first_seed}.npz", **keep)
    json.dump(out, open(f"{args.out_dir}/{args.out_name}.json", "w"), indent=1)
    print("DONE eval", flush=True)


if __name__ == "__main__":
    main()
