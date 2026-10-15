"""Protected-first-phase lr schedule for Exp E's warm-start recipe (docs/debugging/lr-warmup.md).
Same loop as scripts/polaris_scatter_bh_train.sh (init = Exp B ckpt, 1500 scenes x 50 ticks, batch 16, k 4->30, seed 4738, lr cosine to 5%) with an optional
linear lr warmup and a lower peak lr, plus force-error logging at early steps (teacher-forced along truth, expB scenes, bodies with nearest neighbour nn < 1).
Step-1 mechanism probes (control arm): gradient-size stats vs Adam eps, per-module attribution of the first Adam update, counterfactual first updates with larger eps.
Usage: PYTHONPATH=. python scripts/scatter_lrwarm.py --arm ctl|w1|w2 --iters 60769 --init-ckpt B.pt --out-dir D --ckpt-dir C"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as ts
from scripts.scatter_error_growth import DEV, DT, EPS, exact_acc, make_scenes, nn_dist
from scripts.scatter_rootcause import model_fa

VARIANT = "ms_kp_pot_v_g128"
ARMS = {"ctl": (5e-4, 0), "w1": (2.5e-4, 2000), "w2": (1e-4, 2000)}
DIAG_AT = (1, 2, 5, 10, 50, 200, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000)
EPS_CF = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)


@torch.no_grad()
def tf_force(model, scenes, ticks=50):
    e2, a2, nn_all, keep_all = [], [], [], []
    for sc in scenes:
        fa, reset = model_fa(model, sc)
        reset()
        m = sc["m"]
        for t in range(ticks):
            p, v = sc["P"][t], sc["V"][t]
            a = fa(p, v)
            if t < 2:
                continue
            ae = exact_acc(p, m)
            e2.append(((a - ae) ** 2).sum(-1).cpu())
            a2.append((ae ** 2).sum(-1).cpu())
            nn_all.append(nn_dist(p).cpu())
            keep_all.append((p.abs().amax(-1) < 26.0).cpu())
    e2, a2, nn, keep = torch.cat(e2), torch.cat(a2), torch.cat(nn_all), torch.cat(keep_all)
    out = {}
    for name, sel in (("all", keep), ("nn<1", keep & (nn < 1.0)), ("nn>=1", keep & (nn >= 1.0))):
        out[name] = float((e2[sel].mean() / a2[sel].mean()).sqrt()) if sel.any() else None
    return out


def group_of(name):
    return ".".join(name.split(".")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--iters", type=int, default=60769)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--mech", action="store_true", help="step-1 probes (grad stats, per-module attribution, eps counterfactuals)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = ts.EXPS["E"]
    lr, warm = ARMS[args.arm]
    log = lambda s: print(s, flush=True)
    tensors = sf.to_tensors(ts.make_data(cfg, 1500, 50, args.seed, DEV, DT, EPS), DEV)
    model = ts.build(cfg, VARIANT, DT).to(DEV)
    sf.apply_opts(model, "default", "train")
    model.load_state_dict(torch.load(args.init_ckpt, map_location=DEV, weights_only=False)["model"])
    scenes = make_scenes("expB", 12, 9300)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    res = {"arm": args.arm, "peak_lr": lr, "warmup": warm, "iters": args.iters, "seed": args.seed, "force": {}, "mech": {}}
    res["force"]["0"] = tf_force(model, scenes)
    log(f"force @0 {res['force']['0']}")
    path = f"{args.out_dir}/lrwarm_{args.arm}_s{args.seed}.json"

    def force_with(state):
        cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        r = tf_force(model, scenes)
        model.load_state_dict(cur)
        return r

    def mech(model, opt):
        names = [n for n, p in model.named_parameters()]
        params = dict(model.named_parameters())
        grads = {n: params[n].grad.detach().flatten() for n in names if params[n].grad is not None}
        gstats, upd = {}, {}
        for n in names:
            g = grads.get(n)
            if g is None:
                continue
            gr = group_of(n)
            s = gstats.setdefault(gr, {"n": 0, "lt1e-8": 0, "lt1e-6": 0, "gabs_sum": 0.0, "dtheta_over_lr_sum": 0.0, "dnorm2": 0.0})
            d = (params[n].detach() - base[n]).flatten()
            s["n"] += g.numel()
            s["lt1e-8"] += int((g.abs() < 1e-8).sum())
            s["lt1e-6"] += int((g.abs() < 1e-6).sum())
            s["gabs_sum"] += float(g.abs().sum())
            s["dtheta_over_lr_sum"] += float(d.abs().sum()) / opt.param_groups[0]["lr"]
            s["dnorm2"] += float((d ** 2).sum())
            upd[n] = d
        for s in gstats.values():
            s["frac_lt1e-8"] = s["lt1e-8"] / s["n"]
            s["frac_lt1e-6"] = s["lt1e-6"] / s["n"]
            s["mean_abs_grad"] = s["gabs_sum"] / s["n"]
            s["mean_abs_step_over_lr"] = s["dtheta_over_lr_sum"] / s["n"]
            s["update_l2"] = s["dnorm2"] ** 0.5
        res["mech"]["grad_stats_step1"] = gstats
        post = {k: v.detach().clone() for k, v in model.state_dict().items()}
        groups = sorted({group_of(n) for n in names})
        attr = {}
        for gr in groups:
            st = {k: (post[k] if group_of(k) == gr else base[k]) for k in post}
            attr[gr] = force_with(st)["nn<1"]
            log(f"attribution step1 update of {gr} only: nn<1 force err {attr[gr]}")
        res["mech"]["attribution_step1_only_group_updated_nn<1"] = attr
        cf = {}
        for e in EPS_CF:
            st = dict(base)
            for n in names:
                if n in grads:
                    g = grads[n].reshape(params[n].shape)
                    st[n] = base[n] - 5e-4 * g / (g.abs() + e)
            cf[str(e)] = force_with(st)["nn<1"]
            log(f"counterfactual first Adam step lr 5e-4 eps {e}: nn<1 force err {cf[str(e)]}")
        res["mech"]["counterfactual_eps_step1_lr5e-4_nn<1"] = cf
        res["mech"]["init_param_abs_mean"] = {n: float(base[n].abs().mean()) for n in names}

    def hook(it, model, opt):
        if args.mech and it == 1:
            mech(model, opt)
        if it in DIAG_AT:
            res["force"][str(it)] = tf_force(model, scenes)
            log(f"force @{it} lr {opt.param_groups[0]['lr']:.2e} {res['force'][str(it)]}")
            json.dump(res, open(path, "w"))

    targs = SimpleNamespace(lr=lr, lr_warmup=warm, time_budget=0.0, iters=args.iters, k_start=4, k_end=30, batch=16, steps=50, device=str(DEV), seed=args.seed,
                            log_every=100, opt="default", rloss="sq", long_windows=0, step_hook=hook)
    n_it, secs = ts.train(model, tensors, targs, f"{args.ckpt_dir}/E_{VARIANT}.pt", log)
    res["train_iters"], res["train_seconds"] = n_it, secs
    res["force"]["final"] = tf_force(model, scenes)
    json.dump(res, open(path, "w"))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
