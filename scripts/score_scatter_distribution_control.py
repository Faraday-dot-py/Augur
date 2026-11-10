"""Stage 1 of sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon: score the unchanged control (Exp E) on the distribution metric at tick 300.

Scenes are Exp E's own training distribution (half box, half cluster). The control is rolled once (deterministic) from the unperturbed state; the exact reference is the
perturbed float64 ensemble of scripts/scatter_distribution_ensemble.py. Primary perturbation rel is 1e-4; 1e-5 and 1e-3 are reported as sensitivity only. Seeds 9100/9200/9300
are the pre-registered set; 12000/12100/12200 are the fresh confirmation set. One json per (seed, rel) is written and skipped on rerun.
Usage: PYTHONPATH=. python scripts/score_scatter_distribution_control.py --ckpt ~/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out-dir results/scatter_distribution_control
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch

from scripts import chaos_floor as cf
from scripts import scatter_distribution_ensemble as de
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

SEEDS = (9100, 9200, 9300)
FRESH_SEEDS = (12000, 12100, 12200)
RELS = (1e-4, 1e-5, 1e-3)
PASS_FRAC = 0.80
KILL_FRAC = 0.50


def scenes_for(seed, num, dev):
    data = tsf.make_data(tsf.EXPS["E"], num, 1, seed, dev, 0.1, 0.5)
    return [(d[0][0] - sf.CENTER, d[1][0]) for d in data]


def model_final(model, scenes, steps, dev):
    P, V, M, mask = [t.float() for t in sf.to_tensors([(p[None] + sf.CENTER, v[None]) for p, v in scenes], dev)]
    with torch.no_grad():
        Pm, Vm = sf.rollout(model, P[:, 0], V[:, 0], M, mask, steps)
    return Pm.double().cpu().numpy(), Vm.double().cpu().numpy(), [int(m) for m in mask.sum(1).tolist()]


def run_seed(model, seed, rel, args, dev):
    scenes = scenes_for(seed, args.scenes, dev)
    Pm, Vm, ns = model_final(model, scenes, args.steps, dev)
    rng = np.random.default_rng(seed + 777)
    scores, err = [], {5: [], 20: []}
    for i, (p0, v0) in enumerate(scenes):
        n = ns[i]
        pk = de.perturb(p0, rel, args.members, rng)
        pos = torch.tensor(np.concatenate([p0[None], pk]), dtype=torch.float64)
        vel = torch.tensor(np.broadcast_to(v0, (args.members + 1,) + v0.shape).copy(), dtype=torch.float64)
        ex = de.exact_batch(pos, vel, args.steps, dev, keep=(5, 20))
        for k in (5, 20):
            e = np.linalg.norm(Pm[i, k, :n] - ex[k][0][0].numpy(), axis=-1).mean()
            err[k].append(float(e))
        pf, vf = ex[args.steps][0][1:].numpy(), ex[args.steps][1][1:].numpy()
        ens = [de.profiles(pf[j], vf[j]) for j in range(args.members)]
        ens_prof = (np.stack([e[0] for e in ens]), np.stack([e[1] for e in ens]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores.append(de.score_scene(de.profiles(Pm[i, args.steps, :n], Vm[i, args.steps, :n]), ens_prof))
    res = de.pool(scores)
    res.update(seed=seed, rel=rel, scenes=len(scenes), members=args.members, steps=args.steps, err5=float(np.mean(err[5])), err20=float(np.mean(err[20])))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--scenes", type=int, default=16)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS + FRESH_SEEDS)))
    ap.add_argument("--rels", default=",".join(map(str, RELS)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.steps < 20:
        raise SystemExit("--steps must be at least 20 (err@20 is reported)")
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    rels = [float(r) for r in args.rels.split(",")]
    model = cf.load_model("E", args.ckpt, 0.1, dev)
    allres = []
    for seed in seeds:
        for rel in rels:
            path = f"{args.out_dir}/seed{seed}_rel{rel:g}.json"
            if os.path.exists(path):
                allres.append(json.load(open(path)))
                continue
            r = run_seed(model, seed, rel, args, dev)
            json.dump(r, open(path + ".tmp", "w"), indent=1)
            os.replace(path + ".tmp", path)
            print(f"seed {seed} rel {rel:g} density {r['density']['frac']:.3f} (self {r['density']['self_frac']:.3f}) speed {r['speed']['frac']:.3f} (self {r['speed']['self_frac']:.3f}) err20 {r['err20']:.4g}", flush=True)
            allres.append(r)
    summary = {}
    for name, ss in (("pre_registered", SEEDS), ("fresh", FRESH_SEEDS)):
        rows = [r for r in allres if r["seed"] in ss and r["rel"] == rels[0]]
        if len(rows) != len(ss):
            continue
        worst = min(min(r["density"]["frac"], r["speed"]["frac"]) for r in rows)
        summary[name] = dict(worst_fraction=worst, verdict="CONTROL-PASSES-CLOSED" if worst >= PASS_FRAC else "CONTROL-BELOW-KILL-FRACTION" if worst < KILL_FRAC else "CONTROL-BELOW-PASS-OPEN")
    json.dump(dict(summary=summary, rows=allres, ckpt=args.ckpt, torch=torch.__version__), open(f"{args.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
