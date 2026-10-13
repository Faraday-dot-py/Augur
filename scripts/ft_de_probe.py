"""Wide-sample energy-drift probe for fine-tuned scatter-field checkpoints (the harness nr100/bh300 dE rows are medians over 4 scenes).

Per checkpoint spec: nr100 (N scenes, 1000 ticks, scale-init as Exp B) and bh300 (relativistic c=10 collapse) rolled with the model, dE/(K0+|U0|) at ticks
100/300/1000 as median, p16, p84 and signed median, plus expB err@5/20 over 48 scenes. Scenes are fresh seeds (default 13000+), disjoint from the harness seeds.
Spec: PATH or PATH|ALPHA (weights interpolated base + ALPHA * (PATH - base), base = --base, default Exp E).
Usage: PYTHONPATH=. python scripts/ft_de_probe.py --specs a.pt,b.pt|0.5 --out results/ft_de/probe.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_conservation as sc
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

BASE = "checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
TICKS = (100, 300, 1000)


def load_state(path, dev):
    return torch.load(path, map_location=dev, weights_only=False)["model"]


def build(state, dev, opt):
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    sf.load_compat(model, state)
    model.eval()
    sf.apply_opts(model, opt)
    return model


def scenes(kind, n, count, seed0):
    ps, vs = [], []
    for i in range(count):
        rng = np.random.default_rng(seed0 + i)
        p, v = sc.init_scene(kind, n, rng)
        ps.append(p)
        vs.append(v)
    return np.stack(ps), np.stack(vs)


def cons(model, kind, n, count, seed0, steps, dev, eps, dt):
    pos0, vel0 = scenes(kind, n, count, seed0)
    P0, V0 = torch.tensor(pos0, device=dev), torch.tensor(vel0, device=dev)
    rel = kind == "bh"
    r0 = (P0 - P0.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()
    rec = sc.Recorder(rel, eps, model.extent, r0, steps, count, dev, dt)
    sc.roll_model(model, P0, V0, rel, steps, rec, None, dt)
    r = rec.rows
    scale = r["K"][0] + r["U"][0].abs()
    out = {"finite": float(r["finite"].min(0).values.mean())}
    for t in TICKS:
        d = (r["E"][t] - r["E"][0]) / scale
        a = d.abs().cpu().numpy()
        out[f"dE@{t}"] = {"med": float(np.median(a)), "p16": float(np.percentile(a, 16)), "p84": float(np.percentile(a, 84)),
                          "signed_med": float(d.median()), "frac_nonfinite": float((~torch.isfinite(d)).double().mean())}
    return out


def err_expb(model, dev, eps, dt):
    data = gs.make_dataset(48, (10, 100), 20, 13000, scale=True, device=dev, dt=dt, eps=eps)
    P, V, M, mask = sf.to_tensors(data, dev)
    Pm, Vm = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), 20)
    e = (((Pm.double() - P) ** 2).sum(-1).sqrt() * mask[:, None]).sum(2) / mask.sum(1)[:, None]
    return {"err@5": float(e[:, 5].mean()), "err@20": float(e[:, 20].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--opt", default="default")
    ap.add_argument("--contact", action="store_true", help="specs are option-C checkpoints, evaluated with no geometry (eager)")
    ap.add_argument("--contact-free", action="store_true")
    ap.add_argument("--n-nr", type=int, default=24)
    ap.add_argument("--n-bh", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=13000)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    args = ap.parse_args()
    dev = torch.device("cuda")
    base = load_state(args.base, dev)
    res = {}
    for spec in args.specs.split(","):
        path, _, alpha = spec.partition("|")
        if args.contact:
            from scripts import scatter_contact_scenes as cs

            model = cs.build("C", dev, path)
            model.set_scene(None, None)
            if args.contact_free:
                print("contact free-space: gscale", cs.free_space(model), flush=True)
            r = {"nr100": cons(model, "nr", 100, args.n_nr, args.seed0, 1000, dev, args.eps, args.dt),
                 "bh300": cons(model, "bh", 300, args.n_bh, args.seed0 + 500, 1000, dev, args.eps, args.dt),
                 "expB": err_expb(model, dev, args.eps, args.dt)}
            res[spec] = r
            print(f"{spec}: nr100 dE@1000 med {r['nr100']['dE@1000']['med']:.4g} [{r['nr100']['dE@1000']['p16']:.3g},{r['nr100']['dE@1000']['p84']:.3g}] "
                  f"dE@100 {r['nr100']['dE@100']['med']:.4g} | bh300 dE@100 {r['bh300']['dE@100']['med']:.4g} dE@1000 {r['bh300']['dE@1000']['med']:.4g} | "
                  f"expB err@5 {r['expB']['err@5']:.4g} err@20 {r['expB']['err@20']:.4g}", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
            continue
        state = load_state(path, dev)
        if alpha:
            a = float(alpha)
            state = {k: (base[k] + a * (v - base[k])) if v.dtype.is_floating_point else v for k, v in state.items()}
        model = build(state, dev, args.opt)
        r = {"nr100": cons(model, "nr", 100, args.n_nr, args.seed0, 1000, dev, args.eps, args.dt),
             "bh300": cons(model, "bh", 300, args.n_bh, args.seed0 + 500, 1000, dev, args.eps, args.dt),
             "expB": err_expb(model, dev, args.eps, args.dt)}
        res[spec] = r
        print(f"{spec}: nr100 dE@1000 med {r['nr100']['dE@1000']['med']:.4g} [{r['nr100']['dE@1000']['p16']:.3g},{r['nr100']['dE@1000']['p84']:.3g}] "
              f"dE@100 {r['nr100']['dE@100']['med']:.4g} | bh300 dE@100 {r['bh300']['dE@100']['med']:.4g} dE@1000 {r['bh300']['dE@1000']['med']:.4g} | "
              f"expB err@5 {r['expB']['err@5']:.4g} err@20 {r['expB']['err@20']:.4g}", flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
