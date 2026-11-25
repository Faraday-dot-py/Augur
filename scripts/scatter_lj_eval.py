"""Scatter field on the six lj_sim truth scenes, same metrics as scripts/lj_eval.py (T, psi6, P, g(r), E/N over the last --tail frames,
position error vs the truth-rerun chaos floor at frames 1,2,5,10,20,50,100). Writes results/scatter_lj_<tag>.json and the rollout npz files
(lj_sim layout) for rendering. Usage: PYTHONPATH=. python -m scripts.scatter_lj_eval --ckpt <pt|zeroshot> --tag X [--frames 800]"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import lj_eval
from scripts import lj_sim
from scripts import scatter_lj as sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="finetuned checkpoint, or an Exp E checkpoint with --zeroshot")
    ap.add_argument("--zeroshot", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenes", default=",".join(lj_sim.SCENES))
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--floor-frames", type=int, default=100)
    ap.add_argument("--tail", type=int, default=200)
    ap.add_argument("--prefix", default="results/lj")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if args.zeroshot:
        model = sl.build_model(args.ckpt, dev)
    else:
        model = sl.load_model(args.ckpt, dev)
    model.eval()
    out = {"ckpt": args.ckpt, "zeroshot": args.zeroshot}
    d = np.linspace(0.75, 2.5, 100)
    full, pair = sl.two_body_force(model, d, dev)
    _, f_rc = lj_sim._sf_consts(lj_sim.RC)
    ft = 24 * (2 * d ** -13 - d ** -7) - f_rc
    out["force"] = {"d": d.tolist(), "model_full": full.tolist(), "model_pair_term": pair.tolist(), "true": ft.tolist(),
                    "max_abs_tail_1.2_2.5": float(np.abs(full - ft)[d > 1.2].max()), "max_abs_core_0.75_1.2": float(np.abs(full - ft)[d <= 1.2].max())}
    for name in args.scenes.split(","):
        truth = np.load(f"{args.prefix}_{name}.npz")
        rec = int(round(truth["stats"][1, 0] - truth["stats"][0, 0]))
        box = truth["box"]
        ramp = lj_sim.SCENES[name].get("ramp")
        pos, vel = truth["pos"][0].astype(np.float64), truth["vel"][0].astype(np.float64)
        steps = int(truth["stats"][-1, 0])
        nf = min(args.frames, len(truth["pos"]) - 1)
        fp, fv, st, ok = sl.rollout(model, pos, vel, box, nf, dev, ramp=ramp, total_steps=steps, log=lambda s, n=name: print(n, s, flush=True))
        np.savez_compressed(f"{args.out_dir}/scatter_lj_{args.tag}_{name}.npz", pos=fp, vel=fv, box=box, stats=st, dt=float(truth["dt"]), rho=truth["rho"])
        fl, _, _ = lj_sim.run(pos, vel, box, args.floor_frames * rec, rec, float(truth["dt"]), dev, ramp=ramp)
        mdl = {"pos": fp, "stats": st, "box": box}
        tail = min(args.tail, len(fp) - 1)
        ks = [k for k in (1, 2, 5, 10, 20, 50, 100) if k < len(fp)]
        out[name] = {
            "finite": bool(ok), "frames": len(fp) - 1,
            "err_model": {k: float(lj_eval.err_curve(fp[k], truth["pos"][k], box).mean()) for k in ks},
            "err_rerun_floor": {k: float(lj_eval.err_curve(fl[k], truth["pos"][k], box).mean()) for k in ks if k < len(fl)},
            "truth": lj_eval.summary({"stats": truth["stats"][:len(fp)], "pos": truth["pos"][:len(fp)], "box": box}, dev, tail),
            "model": lj_eval.summary(mdl, dev, tail),
            "model_energy_series": st[::40, 3].tolist(), "truth_energy_series": truth["stats"][:len(fp)][::40, 3].tolist(),
        }
        t, m = out[name]["truth"], out[name]["model"]
        out[name]["g_rel_l2"] = float(np.linalg.norm(np.array(t["g"]) - np.array(m["g"])) / np.linalg.norm(np.array(t["g"])))
        print(name, {k: out[name][k] for k in ("err_model", "err_rerun_floor")}, "T", t["T"], m["T"], "psi6", t["psi6"], m["psi6"], "P", t["P"], m["P"], flush=True)
    json.dump(out, open(f"{args.out_dir}/scatter_lj_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
