"""Fine-tune the scatter-field Exp E checkpoint for one review fix at a time (decisions #4/#6 in docs/debugging/scatter-review-decisions-2026-10-06.md).

--fix ctrl     control: E-style scenes only, same optimiser/budget (separates "more training" from the fix)
--fix pairctx  same scenes as ctrl, with --variant ms_kp_pot_v_g128_pairctx (neighbour-context pair branch)
--fix cu       cell-unit kernel/pair features (variant *_cu) + random cell size h in {0.25, 0.5, 1.0} (grid 256/128/64, extent 64)
--fix largen   dense large-N clusters (N 1000-6000) mixed with E scenes
--fix mass     log-uniform masses 0.1-10 (per-body spectrum, equal-mass scale, one dominant body; 30% unit-mass)
--fix masswide same with 0.01-100 (init from the mass run)
--fix eps      softening eps log-uniform groups 0.1-2 and an eps input (variant *_eps)
--fix dense    sigma 1-5 clusters and Plummer a 0.25-2 mixed with E scenes
Variants with new opt-in inputs (cu, eps, mnorm, dnorm) warm-start from E via scatter_field.load_compat.
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_ft_data as fd
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

VAR = "ms_kp_pot_v_g128"
EPS_SET = (0.1, 0.2, 0.35, 0.5, 1.0, 2.0)


def group(name, data, dev, weight, batch, kmax, setup=None, cap=True):
    return dict(name=name, t=[t.float() for t in sf.to_tensors(data, dev)], weight=weight, batch=batch, kmax=kmax, setup=setup, cap=cap)


def set_h(grid):
    return lambda m: ops.retarget(m, grid, 64.0)


def set_eps(e):
    def f(m):
        m.eps_cur = e
    return f


def build_groups(args, dev, log):
    s, n = args.steps, args.scenes
    t0 = time.time()
    g = []
    if args.fix in ("ctrl", "pairctx"):
        g.append(group("base", fd.base(n, s, args.seed, dev), dev, 1.0, args.batch, 99))
    elif args.fix == "cu":
        first = group("base_G256", fd.base(n, s, args.seed, dev), dev, 0.3, args.batch, 99, set_h(256))
        g.append(first)
        for grid, w in ((128, 0.4), (64, 0.3)):
            g.append(dict(first, name=f"base_G{grid}", weight=w, setup=set_h(grid)))
    elif args.fix == "largen":
        g.append(group("base", fd.base(n, s, args.seed, dev), dev, 0.4, args.batch, 99))
        g.append(group("N1000-2500", fd.large_n(max(24, n // 25), s, args.seed + 7, dev, (1000, 2500)), dev, 0.4, 2, 12, cap=False))
        g.append(group("N3000-6000", fd.large_n(max(12, n // 50), s, args.seed + 8, dev, (3000, 6000)), dev, 0.2, 1, 8, cap=False))
    elif args.fix in ("mass", "massnorm"):
        g.append(group("mass0.1-10", fd.mass_spectrum(n, s, args.seed, dev, 0.1, 10.0), dev, 1.0, args.batch, 99))
    elif args.fix == "masswide":
        g.append(group("mass0.01-100", fd.mass_spectrum(n, s, args.seed, dev, 0.01, 100.0), dev, 1.0, args.batch, 99))
    elif args.fix == "eps":
        for e in EPS_SET:
            g.append(group(f"eps{e}", fd.eps_scenes(max(60, n // len(EPS_SET)), s, args.seed + int(e * 100), dev, e), dev, 1.0, args.batch, 10 if e <= 0.2 else 99, set_eps(e)))
    elif args.fix == "dense":
        g.append(group("dense", fd.dense(n, s, args.seed, dev), dev, 1.0, args.batch, 99))
    for x in g:
        log(f"group {x['name']}: scenes {x['t'][0].shape[0]} N {x['t'][0].shape[2]} weight {x['weight']} batch {x['batch']}")
    log(f"data generation {time.time() - t0:.0f}s")
    return g


STOP = {"flag": False}


def on_term(signum, frame):
    STOP["flag"] = True


def val_loss(model, vdata, dev, k):
    P, V, M, mask = vdata
    with torch.no_grad():
        pos, vel, m = P[:, 0], V[:, 0], M
        field = model.init_field(P.shape[0], dev)
        lp = lv = 0.0
        w = mask[:, :, None]
        denom = mask.sum() * 2 * k
        for s in range(k):
            pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mask, field)
            lp = lp + (((pos - P[:, s + 1]) ** 2) * w).sum() / denom
            lv = lv + (((vel - V[:, s + 1]) ** 2) * w).sum() / denom
    return float(lp + 0.1 * lv)


def train(model, groups, args, ckpt_path, log, vdata=None):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, vals = 0, []
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    if os.path.exists(ckpt_path):
        if args.iters and not args.resume:
            raise SystemExit(f"{ckpt_path} exists; pass --resume to continue or remove it")
        st = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["it"]
        vals = st.get("val", [])
        if "rng" in st:
            g.set_state(st["rng"])
        else:
            g.manual_seed(args.seed + start)
        log(f"resumed from it {start}")
    val_every = max(1, args.iters // 40) if args.iters else 0

    def save(it_):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it_, "rng": g.get_state(), "val": vals}, ckpt_path)
    wts = torch.tensor([x["weight"] for x in groups])
    skipped = 0
    t_start = time.time()
    it = start - 1
    last = {}
    while True:
        it += 1
        if args.iters:
            if it >= args.iters:
                break
            frac = it / args.iters
        else:
            frac = (time.time() - t_start) / args.time_budget
            if frac >= 1.0:
                break
        gi = int(torch.multinomial(wts, 1, generator=g))
        grp = groups[gi]
        if grp["setup"]:
            grp["setup"](model)
        P, V, M, mask = grp["t"]
        S, B = P.shape[0], grp["batch"]
        k = min(int(round(args.k_start + (args.k_end - args.k_start) * frac)), grp["kmax"])
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac))) * min(1.0, (it - start + 1) / args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(S, (B,), generator=g).to(args.device)
        t0 = torch.randint(0, args.steps - k + 1, (B,), generator=g).to(args.device)
        tt = t0[:, None] + torch.arange(k + 1, device=args.device)[None]
        Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
        m, mk = M[idx], mask[idx]
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = model.init_field(B, args.device)
        lp = lv = 0.0
        w = mk[:, :, None]
        denom = mk.sum() * 2 * k
        for s in range(k):
            pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
        loss = lp + 0.1 * lv
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn) and (not grp["cap"] or args.loss_cap <= 0 or loss.item() < args.loss_cap):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} {grp['name']} loss {loss.item():.3g} non-finite or above cap, step skipped ({skipped} total)")
        last.setdefault(grp["name"], []).append(loss.item())
        if vdata is not None and val_every and it % val_every == 0:
            vals.append((it, val_loss(model, vdata, args.device, args.val_k)))
            log(f"it {it} validation {vals[-1][1]:.6f}")
        if STOP["flag"]:
            if all(torch.isfinite(p).all() for p in model.parameters()):
                save(it + 1)
            log(f"SIGTERM at it {it}, checkpoint saved")
            sys.exit(143)
        if it % args.log_every == 0:
            msg = " ".join(f"{k_}:{np.mean(v[-20:]):.5f}" for k_, v in last.items())
            log(f"it {it} k {k} loss {loss.item():.6f} [{msg}] t {time.time() - t_start:.0f}s")
            if all(torch.isfinite(p).all() for p in model.parameters()):
                save(it + 1)
    if all(torch.isfinite(p).all() for p in model.parameters()):
        save(it)
    else:
        raise SystemExit("non-finite parameters at end of training, final checkpoint NOT written")
    log(f"trained {it} iters in {time.time() - t_start:.0f}s")
    return it, time.time() - t_start, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", required=True, choices=["ctrl", "pairctx", "cu", "largen", "mass", "massnorm", "masswide", "eps", "dense"])
    ap.add_argument("--variant", default=VAR)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--scenes", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--loss-cap", type=float, default=3.0)
    ap.add_argument("--time-budget", type=float, default=2400.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--iters", type=int, default=0, help="fixed iteration count (replaces --time-budget when set)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val-scenes", type=int, default=32)
    ap.add_argument("--val-seed", type=int, default=9000)
    ap.add_argument("--val-k", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/scatter_s46")
    ap.add_argument("--ckpt-dir", default="checkpoints/scatter_s46")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    def log(s):
        print(s, flush=True)

    model = tsf.build(tsf.EXPS["E"], args.variant, 0.1).to(dev)
    sf.load_compat(model, torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
    log(f"=== {args.name} fix {args.fix} variant {args.variant} params {sum(p.numel() for p in model.parameters())} init {args.init_ckpt}")
    assert (args.fix == "pairctx") == args.variant.endswith("_pairctx"), "fix pairctx needs a *_pairctx variant and vice versa"
    signal.signal(signal.SIGTERM, on_term)
    groups = build_groups(args, dev, log)
    vdata = None
    if args.iters:
        vdata = tuple(t.float() for t in sf.to_tensors(fd.base(args.val_scenes, args.steps, args.val_seed, dev), dev))
    ckpt = f"{args.ckpt_dir}/{args.name}.pt"
    n_it, secs, vals = train(model, groups, args, ckpt, log, vdata)
    json.dump({"name": args.name, "fix": args.fix, "variant": args.variant, "iters": n_it, "seconds": secs, "validation": vals, "args": vars(args)},
              open(f"{args.out_dir}/{args.name}_train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
