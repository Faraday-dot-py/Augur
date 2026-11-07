"""Moon-failure-cause training arms: the solar-scale-translation-augmentation-more-moon-situations recipe (256 cached train-more-moon-situations scenes, init orbit_ft2, k 4->30, 6000 its, translate amp 4 limit 30, kpad ext, seed 4738) with one change per arm.

  --loss-mode moon-weighted   moon bodies weigh --moon-weight (20) in the position and velocity terms; weighted mean (denominator = sum of weights)
  --loss-mode relative-error  position error^2 divided by r^2 (moon: distance to host at window start; planet and belt: distance to the star centre; star: 1), r clamped at 0.1, rescaled by the batch mean r^2 of real bodies so the average weight stays 1; velocity term unchanged
  --curriculum-stages 12      n-ramp staircase (train_scatter_solar_stability.staircase_at): star, planets and moons throughout, belt fraction 0 -> 1
  --epochs 150                18000 its (k ramps 4 -> 30 over 18000), with --snapshot-every 1000
Needs ckpt/solar_more_moons/scenes_256_0_train-more-moon-situations.pt and ckpt/solar_moon_arms/scene_kinds_256.pt (build_moon_scene_kinds.py).
"""
import argparse
import json
import os
import signal

import torch

from scripts import scatter_field as sf
from scripts import scatter_solar_generalized as gen
from scripts import train_scatter_solar_stability as tr

SCENES = "ckpt/solar_more_moons/scenes_256_0_train-more-moon-situations.pt"
KINDS = "ckpt/solar_moon_arms/scene_kinds_256.pt"


def load_data(dev):
    d = torch.load(SCENES, map_location=dev, weights_only=False)
    k = torch.load(KINDS, map_location=dev, weights_only=False)
    K, H = k["K"], k["H"]
    isb = (K == 3).long()
    return d["P"], d["V"], d["M"], K, H, isb.cumsum(1) - 1, isb.sum(1).to(d["P"].dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--loss-mode", choices=["default", "moon-weighted", "relative-error"], default="default")
    ap.add_argument("--moon-weight", type=float, default=20.0)
    ap.add_argument("--curriculum-stages", type=int, default=0)
    ap.add_argument("--ramp", default=None, help="curriculum regime: comma list of k (rollout steps), n (belt bodies), m (mass-range admission) ramped over 12 stages, or none (all full from the start); see train_scatter_solar_stability.regime_at")
    ap.add_argument("--snapshot-every", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--epoch-iters", type=int, default=120)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--spike-factor", type=float, default=0.0)
    ap.add_argument("--kpad", choices=["zero", "ext"], default="ext")
    ap.add_argument("--translate-amp", type=float, default=4.0)
    ap.add_argument("--translate-limit", type=float, default=30.0)
    ap.add_argument("--opt", default="default,-graphtrain")
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/solar_moon_diagnosis")
    ap.add_argument("--ckpt-dir", default="ckpt/solar_moon_arms")
    args = ap.parse_args()
    args.iters = args.epochs * args.epoch_iters
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    signal.signal(signal.SIGTERM, lambda *_: tr.STOP.__setitem__(0, True))
    log = lambda s: print(s, flush=True)
    model = tr.build_model(args.init_ckpt, dev, args.kpad)
    sf.apply_opts(model, args.opt, "train")
    model.kpad = args.kpad
    model.train()
    log(f"=== {args.arm} params {sum(p.numel() for p in model.parameters())} init {args.init_ckpt} iters {args.iters} k {args.k_start}->{args.k_end} loss {args.loss_mode} curriculum {args.curriculum_stages} batch {args.batch} kpad {model.kpad} opt {sorted(model.opt)}")
    if args.curriculum_stages:
        log("staircase (first it, k, belt fraction): " + json.dumps(tr.staircase_schedule(args.iters, args.k_start, args.k_end, args.curriculum_stages)))
    data = load_data(dev)
    if args.ramp is not None:
        ramp, n = tr.parse_ramp(args.ramp), data[0].shape[0]
        sch, prev = [], None
        for i in range(args.iters):
            cur = tr.regime_at(i, args.iters, ramp, 12, args.k_start, args.k_end, n)
            if cur != prev:
                sch.append((i, *cur))
                prev = cur
        log(f"regime {args.ramp} schedule (first it, k, belt fraction, scenes admitted of {n}): " + json.dumps(sch))
        if "m" in ramp:
            order = tr.mass_rank(data[2])
            moons = (data[3][order] == 2).any(1)
            log("scenes with a moon among the admitted prefix: " + json.dumps({a: int(moons[:a].sum()) for a in sorted({x[3] for x in sch})}))
    log(f"data P {tuple(data[0].shape)} moons {int((data[3] == 2).sum())} belt {int((data[3] == 3).sum())}")
    ckpt = f"{args.ckpt_dir}/{args.arm}.pt"
    it, skipped, secs = tr.train(model, data, args, ckpt, log)
    if it >= args.iters:
        json.dump({"arm": args.arm, "iters": it, "skipped": skipped, "seconds": secs, "ckpt": ckpt, "args": vars(args)}, open(f"{args.out_dir}/{args.arm}_train.json", "w"), indent=1)
        log(f"DONE {args.arm}")


if __name__ == "__main__":
    main()
