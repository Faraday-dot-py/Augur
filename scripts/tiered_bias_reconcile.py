"""Run both tier-0 bias tests on the same checkpoints: this branch's run_bias (near_rel, slope; 18-unit window in 128-box multi-clump
scenes) and the hole-4 test of scripts/tiered_eval.g_bias from branch worktree-agent-a8b09cd8f67d9fd44 (commit 392b7d3; needs its
scripts/tiered_{iface,scenes,truth,adapters,eval}.py next to this file). --arms name=dir pairs, dir holds tiered_J.pt (key tier0);
name=baseline uses the unmodified Exp E tier 0."""
import argparse
import json
import os
import types

import torch

from scripts import tiered_adapters as ta
from scripts import tiered_eval as te
from scripts import train_tiered as ttr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0", default=os.path.expanduser("~/E_ms_kp_pot_v_g128.pt"))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--eval-scenes", type=int, default=24)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    for spec in args.arms.split(","):
        name, d = spec.split("=")
        t0 = ttr.load_tier0(args.tier0, dev)
        if name != "baseline":
            t0.model.load_state_dict(torch.load(os.path.join(d, "tiered_J.pt"), map_location=dev, weights_only=False)["tier0"])
        ad = ta.Tier0Only(t0, torch.float32, ttr.DT, 128.0, name)
        res[name] = {"hole4": {}, "mine": {}}
        for sd in ttr.EVAL_SEEDS:
            b = te.g_bias(ad, dev, sd, False)
            res[name]["hole4"][str(sd)] = {k: b[k] for k in ("windows", "rel_rms_force_err", "mean_signed_diff", "se", "z", "mean_diff_along_far_dir")}
            print(name, "hole4", sd, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name]["hole4"][str(sd)].items()}), flush=True)
        if name != "baseline":
            ns = types.SimpleNamespace(tier0=args.tier0, arms="J", ckpt_dir=d, out=os.path.join(d, "bias_tmp.json"), eval_scenes=args.eval_scenes, eval_steps=args.eval_steps)
            ttr.run_bias(ns, dev, print)
            res[name]["mine"] = json.load(open(ns.out))["J"]
            if "baseline" not in res or not res["baseline"]["mine"]:
                res.setdefault("baseline", {"hole4": {}, "mine": {}})["mine"] = json.load(open(ns.out))["baseline"]
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
