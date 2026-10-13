"""Per-channel / per-neighbour-class eval of a scatter-field checkpoint on held-out seeds (recurrent-state ablation).
Scene sets per seed: expB (10-100 unit-mass bodies, scale-init) and expE (Exp E mix: 50% uniform 10-200, 50% cold clusters 80-300), 100 ticks.
Reports err@5/10/20/50/100 (mean per-body position error), rms error per channel (pos x/y, vel x/y) at 5/10/20, model vs truth energy drift
and momentum drift at 100, and err@20/@100 split by tercile of the body's initial nearest-neighbour distance (all masses are 1).
Usage: PYTHONPATH=. python scripts/scatter_recurrent_ablation_eval.py --ckpt <ckpt> --variant ms_kp_pot_v_g128_norec --out results/recabl/norec_eval.json"""
import argparse
import json

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

SEEDS = (9100, 9200, 9300)
TICKS = (5, 10, 20, 50, 100)


def nn_tercile_err(Pm, P, mask):
    out = {}
    d = (P[:, 0, :, None] - P[:, 0, None]).norm(dim=-1) + torch.eye(P.shape[2], dtype=P.dtype)[None] * 1e9 + (1 - mask)[:, None] * 1e9
    nn = d.min(-1).values
    e = (Pm - P).norm(dim=-1)
    vals = nn[mask > 0]
    q = torch.quantile(vals, torch.tensor([1 / 3, 2 / 3], dtype=vals.dtype))
    cls = (nn > q[0]).long() + (nn > q[1]).long()
    for c, name in enumerate(("near", "mid", "far")):
        sel = (cls == c) & (mask > 0)
        out[name] = {**{f"err@{t}": float(e[:, t][sel].mean()) for t in (20, 100)}, "n": int(sel.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--opt", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    dt, eps = 0.1, 0.5
    model = tsf.build(tsf.EXPS["E"], args.variant, dt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    for o in [x for x in args.opt.split(",") if x]:
        sf.apply_opt(model, o)
    res = {"ckpt": args.ckpt, "variant": args.variant, "recurrent": model.recurrent, "seeds": list(SEEDS), "sets": {}}
    for setname in ("expB", "expE"):
        per = {}
        for seed in SEEDS:
            if setname == "expB":
                data = gs.make_dataset(args.scenes, (10, 100), 100, seed, scale=True, device=dev, dt=dt, eps=eps)
            else:
                data = tsf.make_data(tsf.EXPS["E"], args.scenes, 100, seed, dev, dt, eps)
            m, (Pm, Vm, P, V) = tsf.evaluate_model(model, data, dev, dt, eps)
            _, _, _, mask = sf.to_tensors(data, dev)
            per[str(seed)] = {
                "err": {str(t): m["err"][t - 1] for t in TICKS},
                "chan_rms_at_5_10_20": m["chan_rms_at_5_10_20"],
                "dE_rel_model_100": m["energy_drift_rel_model"][100], "dE_rel_true_100": m["energy_drift_rel_true"][100],
                "dP_model_100": m["mom_drift_model"][100], "dP_true_100": m["mom_drift_true"][100],
                "frac_in_grid_final": m["frac_in_grid_final"], "const_vel_err": {str(t): m["const_vel_err"][t - 1] for t in TICKS},
                "nn_terciles": nn_tercile_err(Pm.cpu(), P.cpu(), mask.cpu()),
            }
            print(setname, seed, per[str(seed)]["err"], flush=True)
        keys = ("err", "const_vel_err")
        agg = {k: {t: float(np.mean([per[s][k][t] for s in per])) for t in per[str(SEEDS[0])][k]} for k in keys}
        for k in ("dE_rel_model_100", "dE_rel_true_100", "dP_model_100", "dP_true_100", "frac_in_grid_final"):
            agg[k] = float(np.mean([per[s][k] for s in per]))
        agg["chan_rms_at_5_10_20"] = {c: [float(np.mean([per[s]["chan_rms_at_5_10_20"][c][i] for s in per])) for i in range(3)]
                                      for c in per[str(SEEDS[0])]["chan_rms_at_5_10_20"]}
        agg["nn_terciles"] = {c: {k: float(np.mean([per[s]["nn_terciles"][c][k] for s in per])) for k in ("err@20", "err@100")}
                              for c in ("near", "mid", "far")}
        res["sets"][setname] = {"per_seed": per, "mean": agg}
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
