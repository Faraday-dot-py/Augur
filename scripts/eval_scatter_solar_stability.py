"""Evaluate solar-scale orbit-stability checkpoints against the frozen orbit_ft2 baseline on identical solar initial conditions, edge fix on (--kpad ext).

  python scripts/eval_scatter_solar_stability.py --baseline orbit_ft2.pt --arm solar-scale-long-horizon=ckpt.pt [--arm ...]

Seeds: the showcase IC (4738) first, selection seeds 9100/9200 (pick the arm), confirmation seeds 9300/12000 (pass/kill is judged here only).
Planet survival uses the scatter_solar.solar_metrics rule (not lost = radius within 3x truth max, finite, inside |x| < 32).
Verdict (pre-registered): PASS = selected arm keeps mean >= 6/8 planets at tick 2000 on the confirmation seeds and beats the baseline there;
KILL = mean < 4/8 or no better than the baseline on the same seeds; otherwise INCONCLUSIVE.

Second mode, demo-only and not a research result (solar-scale-state-assimilation-exact-integrator-correction): the selected model is rolled with
every N ticks (default 50 and 100) each body moved a fraction (default 0.1) of the way to the exact fp64 integrator state from the same initial condition.
"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import scatter_solar as sol
from scripts import train_scatter_solar_stability as tr

TICKS = [100, 300, 1000, 2000]
N_PLANETS = 8


def load(ckpt, dev, kpad="ext"):
    return tr.build_model(ckpt, dev, kpad).eval()


def assimilate(pos, vel, p_exact, v_exact, frac):
    return pos + frac * (p_exact - pos), vel + frac * (v_exact - vel)


@torch.no_grad()
def rollout_assimilated(model, scene, Pt, Vt, steps, every, frac, dev):
    """Single-scene rollout; at ticks that are multiples of `every` the state moves `frac` of the way to the exact trajectory (Pt, Vt: (T+1, N, 2) numpy)."""
    pos = torch.tensor(scene["pos"], dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(scene["vel"], dtype=torch.float32, device=dev)[None]
    mass = torch.tensor(scene["mass"], dtype=torch.float32, device=dev)[None]
    mask = torch.ones_like(mass)
    Pe = torch.tensor(Pt, dtype=torch.float32, device=dev)
    Ve = torch.tensor(Vt, dtype=torch.float32, device=dev)
    field = model.init_field(1, dev, pos.dtype)
    ps, vs = [pos], [vel]
    for t in range(1, steps + 1):
        pos, vel, mass, field, _ = sf._step_m(model, pos, vel, mass, mask, field)
        if every > 0 and t % every == 0:
            pos, vel = assimilate(pos, vel, Pe[t][None], Ve[t][None], frac)
        ps.append(pos)
        vs.append(vel)
    return torch.stack(ps, 1)[0].double().cpu().numpy(), torch.stack(vs, 1)[0].double().cpu().numpy()


def survivors(metrics, tick=2000):
    return N_PLANETS * (1.0 - metrics["planets"][str(tick)]["lost_frac"])


def brief(metrics):
    return {"survivors": {str(t): survivors(metrics, t) for t in TICKS},
            "period_err_median": {str(t): metrics["planets"][str(t)]["period_err_median"] for t in TICKS},
            "belt_lost_frac_2000": metrics["belt"]["2000"]["lost_frac"],
            "moon_bound_at_T": metrics["moon"]["bound_at_T_model"], "moon_bound_at_T_truth": metrics["moon"]["bound_at_T_truth"]}


def mean_surv(table, name, seeds):
    return float(np.mean([table[str(s)][name]["survivors"]["2000"] for s in seeds]))


def verdict(table, arm, baseline, confirm):
    a, b = mean_surv(table, arm, confirm), mean_surv(table, baseline, confirm)
    if a < 4.0 or a <= b:
        return "KILL", a, b
    return ("PASS" if a >= 6.0 else "INCONCLUSIVE"), a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--baseline-name", default="orbit_ft2")
    ap.add_argument("--arm", action="append", default=[], help="name=ckpt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--first-seed", type=int, default=4738)
    ap.add_argument("--select-seeds", default="9100,9200")
    ap.add_argument("--confirm-seeds", default="9300,12000")
    ap.add_argument("--assimilate-every", default="50,100")
    ap.add_argument("--assimilate-frac", type=float, default=0.1)
    ap.add_argument("--belt", type=int, default=200)
    ap.add_argument("--kpad", choices=["zero", "ext"], default="ext")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/solar_stability")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    sel = [int(x) for x in args.select_seeds.split(",")]
    conf = [int(x) for x in args.confirm_seeds.split(",")]
    seeds = [args.first_seed] + sel + conf
    ckpts = {args.baseline_name: args.baseline, **dict(a.split("=", 1) for a in args.arm)}
    models = {n: load(c, dev, args.kpad) for n, c in ckpts.items()}
    table, truths, trajs = {}, {}, {}
    for seed in seeds:
        scene = sol.solar_scene(seed, args.belt, tr.HILL_K)
        Pt, Vt = ss.nbody_truth(scene["pos"], scene["vel"], scene["mass"], args.steps, dev)
        truths[seed] = (scene, Pt, Vt)
        table[str(seed)], trajs[seed] = {}, {"truth": Pt.astype(np.float32), "mass": scene["mass"], "kind": scene["kind"]}
        for name, model in models.items():
            Pm, Vm, _ = ss.model_rollout(model, [scene], args.steps, dev)
            met = sol.solar_metrics(Pm[0], Vm[0], Pt, Vt, scene, TICKS)
            table[str(seed)][name] = brief(met)
            trajs[seed]["model_" + name] = Pm[0].astype(np.float32)
            print(seed, name, json.dumps(table[str(seed)][name]["survivors"]), json.dumps({k: round(v, 3) for k, v in table[str(seed)][name]["period_err_median"].items()}), flush=True)
    arms = [n for n in models if n != args.baseline_name]
    out = {"seeds": {"first": args.first_seed, "select": sel, "confirm": conf}, "table": table, "ckpts": ckpts}
    if arms:
        pick = max(arms, key=lambda n: (mean_surv(table, n, sel), -float(np.mean([table[str(s)][n]["period_err_median"]["2000"] for s in sel]))))
        v, a, b = verdict(table, pick, args.baseline_name, conf)
        out["selected"] = pick
        out["verdict"] = {"arm": pick, "result": v, "mean_survivors_2000_confirm": a, "baseline_mean_survivors_2000_confirm": b,
                          "per_arm_select_mean_2000": {n: mean_surv(table, n, sel) for n in arms},
                          "per_arm_confirm_mean_2000": {n: mean_surv(table, n, conf) for n in arms}}
        print("SELECTED", pick, "VERDICT", json.dumps(out["verdict"]), flush=True)
        out["state_assimilation_demo_only"] = {"label": "demo-only labelled mode, not a research result: exact fp64 integrator state mixed in", "frac": args.assimilate_frac, "model": pick, "runs": {}}
        for every in [int(x) for x in args.assimilate_every.split(",")]:
            tag = f"every{every}"
            out["state_assimilation_demo_only"]["runs"][tag] = {}
            for seed in seeds:
                scene, Pt, Vt = truths[seed]
                Pm, Vm = rollout_assimilated(models[pick], scene, Pt, Vt, args.steps, every, args.assimilate_frac, dev)
                met = sol.solar_metrics(Pm, Vm, Pt, Vt, scene, TICKS)
                out["state_assimilation_demo_only"]["runs"][tag][str(seed)] = brief(met)
                trajs[seed][f"model_{pick}_assimilated_{tag}"] = Pm.astype(np.float32)
                print("assimilated", tag, seed, json.dumps(brief(met)["survivors"]), flush=True)
    for seed in seeds:
        np.savez_compressed(f"{args.out_dir}/traj_seed{seed}.npz", **trajs[seed])
    json.dump(out, open(f"{args.out_dir}/eval_summary.json", "w"), indent=1)
    print("DONE eval", flush=True)


if __name__ == "__main__":
    main()
