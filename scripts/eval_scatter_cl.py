"""Past-horizon behaviour of a scatter-field checkpoint (Exp E eval distribution, seeds 9000/12000, 48 scenes, 100 ticks).

Per tick: err (mean per-body position error vs truth), per-channel rms (pos x/y, vel x/y), field-channel rms/max (recurrent state), model |a| rms/max
(a0 = the carried Verlet acceleration) next to the truth |a| rms/max, number of scenes non-finite; first non-finite tick. Also relabelled err@20 at the
curriculum offsets 0/5/10/20 (closed-loop one-window error). The training horizon is 20 ticks, so ticks 20-100 are past it.

Usage: PYTHONPATH=. python scripts/eval_scatter_cl.py --ckpt <ckpt> --out <json> [--variant ms_kp_pot_v_g128]
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_cl as cl
from scripts import train_scatter_field as tsf

TICKS = (5, 10, 20, 30, 40, 50, 75, 100)


def rollout_stats(model, P, V, M, mask, steps, eps):
    B = P.shape[0]
    Mf, mk = M.float(), mask.float()
    pos, vel, state = P[:, 0].float(), V[:, 0].float(), model.init_field(B, P.device)
    cnt = mask.sum()
    rec = {k: [] for k in ("err", "pos_x", "pos_y", "vel_x", "vel_y", "field_rms", "field_max", "a_rms", "a_max", "true_a_rms", "true_a_max", "nonfinite")}
    with torch.no_grad():
        for t in range(1, steps + 1):
            pos, vel, _, state, _ = sf._step_m(model, pos, vel, Mf, mk, state)
            pd, vd = pos.double(), vel.double()
            fin = torch.isfinite(pd).all(-1).all(-1) & torch.isfinite(vd).all(-1).all(-1)
            rec["nonfinite"].append(int((~fin).sum()))
            dp, dv = (pd - P[:, t]) * mask[..., None], (vd - V[:, t]) * mask[..., None]
            rec["err"].append(float(((dp ** 2).sum(-1).sqrt().sum(1) / mask.sum(1)).mean()))
            for nm, d in (("pos", dp), ("vel", dv)):
                for ax, a in ((0, "x"), (1, "y")):
                    rec[f"{nm}_{a}"].append(float(((d[..., ax] ** 2).sum() / cnt).sqrt()))
            fld = state[0] if isinstance(state, tuple) else state
            rec["field_rms"].append(float(fld.pow(2).mean().sqrt()))
            rec["field_max"].append(float(fld.abs().max()))
            if isinstance(state, tuple) and state[1] is not None:
                am = state[1].double().norm(dim=-1) * mask
                rec["a_rms"].append(float(((am ** 2).sum() / cnt).sqrt()))
                rec["a_max"].append(float(am.max()))
            ta = cl.accel(P[:, t], M, mask, eps).norm(dim=-1)
            rec["true_a_rms"].append(float(((ta ** 2).sum() / cnt).sqrt()))
            rec["true_a_max"].append(float(ta.max()))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", default="E")
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--opt", default="none")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--scenes", type=int, default=48)
    args = ap.parse_args()
    dev = torch.device("cuda")
    cfg = tsf.EXPS[args.exp]
    model = tsf.build(cfg, args.variant, args.dt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    sf.apply_opts(model, args.opt, "infer")
    res = {"ckpt": args.ckpt, "seeds": {}}
    evs = []
    for seed in tsf.EVAL_SEEDS:
        ev = sf.to_tensors(tsf.make_data(cfg, args.scenes, args.steps, seed, dev, args.dt, args.eps), dev)
        evs.append(ev)
        rec = rollout_stats(model, *ev, args.steps, args.eps)
        nf = rec["nonfinite"]
        first = next((i + 1 for i, n in enumerate(nf) if n), None)
        res["seeds"][str(seed)] = {"series": rec, "first_nonfinite_tick": first}
        print(f"seed {seed} err@{list(TICKS)} {[round(rec['err'][t - 1], 5) for t in TICKS]} field_rms@{list(TICKS)} {[round(rec['field_rms'][t - 1], 4) for t in TICKS]} "
              f"a_rms/true {[round(rec['a_rms'][t - 1] / rec['true_a_rms'][t - 1], 3) if rec['a_rms'] else None for t in TICKS]} first nonfinite {first}", flush=True)
    ns = argparse.Namespace(dt=args.dt, eps=args.eps)
    res["relabel_err20"] = {str(s): float(np.mean([cl.eval_err(model, ev, s, 20, ns) for ev in evs])) for s in cl.OFFSETS}
    print("relabelled err@20 by offset", res["relabel_err20"], flush=True)
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
