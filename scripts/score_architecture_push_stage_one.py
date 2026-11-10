"""Scores the three architecture-push stage-1 experiments against the lines in docs/debugging/architecture-push-prereg/<experiment>.md.
Usage: python scripts/score_architecture_push_stage_one.py ladder|law-blind|scale-input --in ... [--val candidate.done --iters N] --out score.json
Lines are copied from the pre-registration; do not edit them here after launch. A ratio is the mean over eval seeds of candidate / reference per seed;
the reference's own seed-to-seed spread, (max - min) / mean, is printed beside it so a ratio inside that spread is read as such.
"""
import argparse
import json


def plateau_falling(val, iters):
    """val: [{it, val_loss}]. True when the mean over the last 10% of iterations is more than 3% below the mean over the preceding 10%."""
    last = [v["val_loss"] for v in val if v["it"] > 0.9 * iters]
    prev = [v["val_loss"] for v in val if 0.8 * iters < v["it"] <= 0.9 * iters]
    if not last or not prev:
        raise ValueError(f"plateau guard needs validation points in both windows (last {len(last)}, preceding {len(prev)}); log more often")
    return sum(last) / len(last) < 0.97 * sum(prev) / len(prev)


def mean(x):
    return sum(x) / len(x)


def spread(x):
    return (max(x) - min(x)) / mean(x)


def paired(cand, ref):
    r = [c / b for c, b in zip(cand, ref)]
    return {"mean": mean(r), "per_seed": r, "reference_noise_floor": spread(ref)}


def guard(out, val, iters):
    if val is None:
        out["plateau_guard"] = "not applicable (no training in this stage)"
        return
    falling = plateau_falling(val, iters)
    out["plateau_guard"] = "FALLING" if falling else "flat"
    if falling and out["verdict"] in ("PASS", "KILL"):
        out["verdict"] = "INCONCLUSIVE-undertrained"


def score_ladder(res):
    ro = next(v for k, v in res.items() if k.startswith("rollout:"))
    out = {"err20_ratio_vs_E": ro["err20_ratio_mean"], "err20_ratio_per_seed": ro["err20_ratio_per_seed"],
           "E_noise_floor": ro["err20_E_noise_floor"], "tiled_rel_l1_diff": res["tiled"]["rel_l1_diff"],
           "exponent_rel_err_pow2": res["fit:E:pow2"]["exponent_rel_err"], "fit_rel_err_E": res["fit:E:pow2"]["fit_rel_err"]}
    ok = out["err20_ratio_vs_E"] <= 1.15 and out["tiled_rel_l1_diff"] <= 0.04 and out["exponent_rel_err_pow2"] <= 0.10
    out["verdict"] = "KILL" if out["err20_ratio_vs_E"] > 1.30 else ("PASS" if ok else "INCONCLUSIVE")
    out["lines"] = {"pass": "err@20 <= 1.15x E AND tiled <= 4% AND pow2 exponent within 10%", "kill": "err@20 > 1.30x E"}
    return out


def score_law_blind(res, val, iters):
    seeds = sorted(res["candidate"]["grav"])
    g = paired([res["candidate"]["grav"][s]["err20"] for s in seeds], [res["E"]["grav"][s]["err20"] for s in seeds])
    out = {"gravity_err20_ratio_vs_E": g, "held_out_force_ratio_vs_comparator": {}, "trained_law_force_ratio_vs_main": {}}
    for case, cmp in (("HELD_pow1.5", "cmp_pow1.5_single"), ("HELD_yuk32", "cmp_yuk32_single")):
        out["held_out_force_ratio_vs_comparator"][case] = paired([res["candidate"][case][s]["force_rel"] for s in seeds],
                                                                 [res[cmp][case][s]["force_rel"] for s in seeds])
    for case in ("grav", "pow1", "pow3", "yuk8", "yuk128", "lj"):
        out["trained_law_force_ratio_vs_main"][case] = paired([res["candidate"][case][s]["force_rel"] for s in seeds],
                                                              [res["main_s4739"][case][s]["force_rel"] for s in seeds])
    held = all(v["mean"] <= 2.0 for v in out["held_out_force_ratio_vs_comparator"].values())
    out["verdict"] = "KILL" if g["mean"] > 1.15 else ("PASS" if g["mean"] <= 1.05 and held else "INCONCLUSIVE")
    out["lines"] = {"pass": "gravity err@20 <= 1.05x E AND other laws within 2x their single-law comparators (held-out pow1.5 and yuk32 only; no comparators exist for the trained laws)",
                    "kill": "gravity err@20 > 1.15x E"}
    guard(out, val, iters)
    return out


def score_scale_input(far, val, iters):
    seeds = sorted(far["shared_256"])
    f = lambda k: [far[k][s]["far_rel"] for s in seeds]
    out = {"far_rel_shared_1024_over_256": paired(f("shared_1024"), f("shared_256")), "far_rel_shared_256_over_control_256": paired(f("shared_256"), f("control_256"))}
    a, b = out["far_rel_shared_1024_over_256"]["mean"], out["far_rel_shared_256_over_control_256"]["mean"]
    out["verdict"] = "KILL" if a > 2.0 or b > 1.15 else ("PASS" if a <= 1.2 and b <= 1.0 else "INCONCLUSIVE")
    out["lines"] = {"pass": "far-field rel error at 1024 <= 1.2x its value at 256 AND its value at 256 <= 1.0x the per-tier control at 256",
                    "kill": "error at 1024 > 2.0x its value at 256 OR value at 256 > 1.15x the control"}
    guard(out, val, iters)
    return out


def load_val(path):
    if path.endswith(".pt"):
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)["val"]
    return json.load(open(path))["val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ladder", "law-blind", "scale-input"])
    ap.add_argument("--in", dest="inp", action="append", required=True, help="ladder and law-blind: one result json; scale-input: shared_256=path, shared_1024=path, control_256=path")
    ap.add_argument("--val", default="", help="candidate val history: a .done json or a checkpoint .pt, each with a val key")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    val = load_val(a.val) if a.val else None
    if a.which == "ladder":
        out = score_ladder(json.load(open(a.inp[0])))
    elif a.which == "law-blind":
        out = score_law_blind(json.load(open(a.inp[0])), val, a.iters)
    else:
        far = {}
        for spec in a.inp:
            n, path = spec.split("=", 1)
            far[n] = json.load(open(path))
        out = score_scale_input(far, val, a.iters)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
