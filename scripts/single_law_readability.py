"""Single-law readability experiments on the job-7256 runner (scripts/single_law_network_sizing.py): two-number-message-bottleneck-single-law
(width-8 model, kernel and pair MLP each pass through a 2-number code) and sparse-code-unet-single-law (width-32 model, top-8 features per cell at every
U-Net block). Pre-registrations: docs/debugging/experiment-log.md entries of the same names.
Frozen controls are the job-7256 checkpoints in --control-dir (stage one); --train-controls retrains them with the candidate (confirmation, extension).
Usage: PYTHONPATH=. python scripts/single_law_readability.py --experiment bottleneck --out-dir results/x --ckpt-dir ckpt --control-dir $HOME/augur-single-law-network-sizing/ckpt
"""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from scripts import scatter_multi as sm
from scripts import single_law_network_sizing as sz
from scripts import single_law_readability_probes as pr

HELD_SEEDS = (9100, 9200, 9300)
FRESH_SEEDS = (12000, 12100, 12200)
VAL_SEED = 9000
VAL_SCENES = 32
RANK_SEED = 9000
RANK_SCENES = 16
PLATEAU_DECLINE = 0.03
PASS_R2, KILL_R2 = 0.95, 0.80
SLOPE_TOL = 0.10
ERR_PASS, ERR_KILL = 1.15, 1.30
FORCE_PASS, FORCE_KILL = 1.10, 1.30
ABL_PASS, ABL_KILL = 5.0, 2.0
BOTTLENECK = 2
SPARSE_K = 8
SPEC = {"bottleneck": {"width": 8, "controls": (8,)}, "sparse-code": {"width": 32, "controls": (8, 32)}}


def log(s):
    sz.log(s)


def builder(experiment, width):
    if experiment == "bottleneck":
        return lambda w, s, d: sz.build_model(w, s, d, bottleneck=BOTTLENECK)
    return lambda w, s, d: sz.build_model(w, s, d, sparse_k=SPARSE_K)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def dump(path, obj):
    json.dump(jsonable(obj), open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)


@torch.no_grad()
def rollout_val(model, data, dev, nr=20):
    Nb = int(data["n"].max())
    P, V = data["P"][:, :, :Nb], data["V"][:, :, :Nb]
    mk = data["mask"][:, :Nb]
    m = torch.ones_like(mk)
    pos, vel = P[:, 0], V[:, 0]
    fld = model.init_field(P.shape[0], dev)
    for s in range(nr):
        pos, vel, _, fld, _ = sz.sf._step_m(model, pos, vel, m, mk, fld)
    d = (pos - P[:, nr]).norm(dim=-1) * mk
    return float((d.sum(1) / mk.sum(1)).mean())


def window_epochs(epochs):
    w = max(1, round(0.1 * epochs))
    return list(range(epochs - 2 * w + 1, epochs - w + 1)), list(range(epochs - w + 1, epochs + 1))


def plateau(val, epochs):
    prev, last = window_epochs(epochs)
    if any(str(e) not in val for e in prev + last):
        return {"decline": None, "undertrained": None}
    a = float(np.mean([val[str(e)] for e in prev]))
    b = float(np.mean([val[str(e)] for e in last]))
    decline = (a - b) / a
    return {"decline": decline, "undertrained": bool(decline > PLATEAU_DECLINE), "val": val}


def mean_over(ev, key):
    return float(np.mean([ev[s][key] for s in ev]))


def rollout20(ev):
    return float(np.mean([ev[s]["rollout_mean_pos_err"]["20"] for s in ev]))


def verdict_bottleneck(cand, ctrl, probes, plat):
    ratio = rollout20(cand) / rollout20(ctrl)
    r2 = probes["pair_r2_code"]
    slope_ok = probes["kernel_slope_rel_diff"] <= SLOPE_TOL
    if r2 >= PASS_R2 and slope_ok and ratio <= ERR_PASS:
        label = "PASS"
    elif r2 < KILL_R2 or ratio > ERR_KILL:
        label = "KILL"
    else:
        label = "INCONCLUSIVE"
    if label != "PASS" and plat["undertrained"]:
        label = "INCONCLUSIVE-undertrained"
    return {"label": label, "err20_ratio_vs_control": ratio, "min_r2": r2, "slope_rel_diff": probes["kernel_slope_rel_diff"]}


def verdict_sparse(cand, ctrl32, ablation, plat):
    ratio = mean_over(cand, "force_rel_l2") / mean_over(ctrl32, "force_rel_l2")
    sets = list(ablation)
    passes = sum(1 for s in sets if ablation[s]["ratio"] >= ABL_PASS and ablation[s]["above_random_max"])
    kills = sum(1 for s in sets if ablation[s]["ratio"] < ABL_KILL)
    need = 2 if len(sets) == 3 else len(sets)
    if ratio <= FORCE_PASS and passes >= need:
        label = "PASS"
    elif ratio > FORCE_KILL or kills >= need:
        label = "KILL"
    else:
        label = "INCONCLUSIVE"
    if label != "PASS" and plat["undertrained"]:
        label = "INCONCLUSIVE-undertrained"
    return {"label": label, "force_ratio_vs_width32": ratio, "sets_passing_ablation": passes, "sets_killing_ablation": kills}


def control_path(width, args):
    if width == 32 and args.ref32_ckpt:
        return args.ref32_ckpt
    return f"{args.control_dir}/width{width}.pt"


def load_control(width, args, dev):
    model = sz.build_model(width, args.seed, dev)
    path = control_path(width, args)
    assert os.path.exists(path), f"missing control checkpoint {path}"
    st = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    return model.eval(), st["it"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=sorted(SPEC), required=True)
    ap.add_argument("--control-dir", default=None)
    ap.add_argument("--ref32-ckpt", default=None, help="width-32 reference checkpoint; default control-dir/width32.pt (job 7256 seed 4738, which spiked)")
    ap.add_argument("--train-controls", action="store_true")
    ap.add_argument("--eval-seeds", default=",".join(str(s) for s in HELD_SEEDS))
    ap.add_argument("--ablation-draws", type=int, default=30)
    sz.train_args(ap)
    args = ap.parse_args()
    assert args.train_controls or args.control_dir, "frozen controls need --control-dir"
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, sz.on_term)
    seeds = [int(s) for s in args.eval_seeds.split(",")]
    spec = SPEC[args.experiment]
    tag = f"{args.experiment}-e{args.epochs}-s{args.seed}"
    t0 = time.time()
    train_data = sz.gen_data(args.seed, 1, args.scenes, args.steps, dev)
    assert train_data["P"].shape[0] == args.scenes, "training scene generation rejected scenes"
    evals = {s: sz.gen_data(s, 2, args.eval_scenes, args.steps, dev) for s in seeds}
    val_data = sz.gen_data(VAL_SEED, 3, VAL_SCENES, args.steps, dev)
    for s, d in evals.items():
        assert d["P"].shape[0] == args.eval_scenes, f"eval seed {s} short"
    log(f"data in {time.time() - t0:.0f}s; experiment {args.experiment} seed {args.seed} epochs {args.epochs} eval seeds {seeds}")
    val_epochs = tuple(e for w in window_epochs(args.epochs) for e in w)
    val_fn = lambda m: rollout_val(m, val_data, dev)

    ctrl_info = {}
    for w in spec["controls"]:
        rp = f"{args.out_dir}/control-{tag}-width{w}.json"
        if os.path.exists(rp):
            ctrl_info[w] = json.load(open(rp))
            continue
        if args.train_controls:
            model, info = sz.train(w, args, dev, train_data, name=f"control-{tag}-width{w}", val_fn=val_fn, val_epochs=val_epochs)
            model.eval()
        else:
            model, it = load_control(w, args, dev)
            info = {"frozen_from": control_path(w, args), "iters": it}
        ev = {str(s): sz.evaluate(model, d, dev, args.steps) for s, d in evals.items()}
        ctrl_info[w] = {"width": w, "train": info, "eval": ev, "plateau": plateau(info.get("validation", {}), args.epochs) if args.train_controls else None}
        if args.experiment == "bottleneck":
            ctrl_info[w]["probes"] = pr.bottleneck_fit(model, dev)
            ctrl_info[w]["unet_share"] = {str(s): pr.unet_share(model, d, dev) for s, d in evals.items()}
        dump(rp, ctrl_info[w])
        log(f"control width {w} " + " ".join(f"{s}:force={ctrl_info[w]['eval'][str(s)]['force_rel_l2']:.5f}" for s in seeds))

    cw = spec["width"]
    cp = f"{args.out_dir}/candidate-{tag}.json"
    if os.path.exists(cp):
        log(f"candidate result exists, skipping: {cp}")
        print("ALL DONE", flush=True)
        return
    model, info = sz.train(cw, args, dev, train_data, build=builder(args.experiment, cw), name=f"candidate-{tag}", val_fn=val_fn, val_epochs=val_epochs)
    model.eval()
    plat = plateau(info["validation"], args.epochs)
    ev = {str(s): sz.evaluate(model, d, dev, args.steps) for s, d in evals.items()}
    res = {"experiment": args.experiment, "seed": args.seed, "epochs": args.epochs, "eval_seeds": seeds, "params": sz.count_params(model),
           "train": info, "eval": ev, "plateau": plat, "args": vars(args)}
    if args.experiment == "bottleneck":
        res["probes"] = pr.bottleneck_fit(model, dev)
        res["unet_share"] = {str(s): pr.unet_share(model, d, dev) for s, d in evals.items()}
        res["verdict"] = verdict_bottleneck(ev, {s: ctrl_info[8]["eval"][s] for s in ev}, res["probes"], plat)
    else:
        rk = f"{args.out_dir}/ranking-{tag}.json"
        if os.path.exists(rk):
            ranked = [tuple(r) for r in json.load(open(rk))["ranked"]]
        else:
            rank_data = sz.gen_data(RANK_SEED, 4, RANK_SCENES, args.steps, dev)
            base, ranked = pr.rank_features(model, rank_data, dev)
            dump(rk, {"baseline_median": base, "ranked": ranked, "seed": RANK_SEED, "scenes": RANK_SCENES})
            log(f"ranked {len(ranked)} features; top {ranked[:5]}")
        freq = pr.activation_frequency(model, val_data, dev)
        res["ablation"] = {str(s): pr.ablation_test(model, ranked, freq, d, dev, draws=args.ablation_draws) for s, d in evals.items()}
        res["verdict"] = verdict_sparse(ev, {s: ctrl_info[32]["eval"][s] for s in ev}, res["ablation"], plat)
        res["vs_width8_force_ratio"] = mean_over(ev, "force_rel_l2") / mean_over(ctrl_info[8]["eval"], "force_rel_l2")
    res["control_rollout20"] = {str(w): rollout20(ctrl_info[w]["eval"]) for w in ctrl_info}
    res["candidate_rollout20"] = rollout20(ev)
    dump(cp, res)
    log("verdict " + json.dumps(jsonable(res["verdict"])))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
