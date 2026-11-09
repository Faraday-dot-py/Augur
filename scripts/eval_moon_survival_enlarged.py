"""Enlarged two-moons eval for the moon failure cause diagnosis: --seeds seeds x j in 0,1 two-moons scenes (scatter_solar_generalized.eval_rng(seed, "two-moons", j) for seeds 4738 .. 4738+seeds-1, same set as the diagnostics agent (tests 1-5), scene rule unchanged, exact fp64 truth, stable over all 2000 ticks).
Per checkpoint and repeat: mean moon survival tick (first tick the moon is farther than the host Hill radius from the host, tick 2000 if never), planets kept at tick 2000, and a 95% bootstrap interval resampling scenes (2000 resamples, rng 4738).
Noise floor (as the diagnostics agent defines it): max of the bootstrap half-width and the range (max - min) of the repeat means.

  python scripts/eval_moon_survival_enlarged.py --ckpt moon-arm=path.pt [--ckpt NAME=path ...] --repeats 3 --out results/solar_moon_diagnosis/x.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_scenes as ss
from scripts import scatter_solar_generalized as gen
from scripts import train_scatter_solar_stability as tr
from scripts.eval_scatter_solar_translation import moon_survival

FAMILY = "two-moons"


def boot(per_scene, n=2000):
    rng = np.random.default_rng(4738)
    sums = np.array([np.sum(x) for x in per_scene], dtype=float)
    cnt = np.array([len(x) for x in per_scene], dtype=float)
    idx = rng.integers(0, len(per_scene), (n, len(per_scene)))
    means = sums[idx].sum(1) / cnt[idx].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path (checkpoint with a 'model' entry)")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=4738)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--kpad", choices=["zero", "ext"], default="ext")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    NS = 2 * args.seeds
    t0, stats, scenes, truth, truthV = time.time(), {}, [], [], []
    for i in range(2 * args.seeds):
        sc, Pt, Vt = gen.make_scene(gen.eval_rng(args.first_seed + i // 2, FAMILY, i % 2), FAMILY, args.steps, args.steps, dev, stats)
        scenes.append(sc)
        truth.append(Pt)
        truthV.append(Vt)
    print(f"{NS} scenes, truth {time.time() - t0:.0f}s, stats {json.dumps(stats)}", flush=True)
    truth_surv = [moon_survival(truth[i], scenes[i]) for i in range(NS)]
    out = {"family": FAMILY, "scenes": NS, "seeds": [args.first_seed, args.first_seed + args.seeds - 1], "j": [0, 1], "n_moons": int(sum(len(x) for x in truth_surv)), "truth_survival_mean": float(np.mean([a for x in truth_surv for a in x])),
           "scene_stats": stats, "models": {}}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model = tr.build_model(path, dev, args.kpad).eval()
        reps = []
        for r in range(args.repeats):
            per, kept, n_pl = [], 0.0, 0
            for b in range(0, NS, args.batch):
                sub = scenes[b:b + args.batch]
                Pm, Vm, _ = ss.model_rollout(model, sub, args.steps, dev)
                for i, sc in enumerate(sub):
                    n = len(sc["mass"])
                    per.append(moon_survival(Pm[i, :, :n], sc))
                    met = gen.family_metrics(Pm[i, :, :n], Vm[i, :, :n], truth[b + i], truthV[b + i], sc, [2000])
                    kept, n_pl = kept + met["planets"]["2000"]["kept"], n_pl + met["n_planets"]
            flat = [a for x in per for a in x]
            reps.append({"moon_survival_mean": float(np.mean(flat)), "ci95": boot(per), "planets_kept_2000": kept / n_pl, "per_scene": per})
            print(name, "repeat", r, "survival", round(reps[-1]["moon_survival_mean"], 1), "ci", [round(x) for x in reps[-1]["ci95"]], "planets kept", round(kept / n_pl, 3), f"{time.time() - t0:.0f}s", flush=True)
        means = [x["moon_survival_mean"] for x in reps]
        out["models"][name] = {"ckpt": path, "repeats": reps, "mean_of_repeats": float(np.mean(means)), "noise_floor_range": float(max(means) - min(means)) if len(means) > 1 else None}
        out["models"][name]["ci95_half_width"] = float((reps[0]["ci95"][1] - reps[0]["ci95"][0]) / 2)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    print("DONE enlarged eval", flush=True)


if __name__ == "__main__":
    main()
