"""Train a tier-0 scatter field (ScatterField + scatter_tier0 extensions) on windowed gravity scenes with a random smooth
external field; target = truth - external (the external acceleration is applied exactly by `extapply`, the net learns the near field).

Usage: PYTHONPATH=. python scripts/train_tier0_ext.py --t0 "ext=6,extapply,taper=8,bnd,halo=8" --init-ckpt E.pt --out-dir results/tier0_ext --tag armA
Resumes from <out-dir>/<tag>.pt automatically; SIGTERM saves a checkpoint and exits 0. --est prints a time estimate and exits.
"""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m
from scripts import scatter_tier0_data as td
from scripts import train_scatter_field as tsf

STOP = []


def build(args, dev):
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", args.dt).to(dev)
    t0m.attach_tier0(model, args.t0)
    sf.apply_opts(model, "default", "train")
    return model


def set_batch_ext(model, ext, idx):
    phi0, g, H = ext
    model.t0.set_external(model, phi0=phi0[idx], g=g[idx], H=H[idx])


def save(path, model, opt, it, args):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "t0": args.t0, "args": vars(args)}, tmp)
    os.replace(tmp, path)


def schedule(args, it, iters):
    frac = it / max(1, iters - 1)
    k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
    lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
    return k, lr


def step_loss(model, P, V, M, mask, ext, idx, k, t0s, dev):
    set_batch_ext(model, ext, idx)
    tt = t0s[:, None] + torch.arange(k + 1, device=dev)[None]
    Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
    m, mk = M[idx], mask[idx]
    pos, vel = Pw[:, 0], Vw[:, 0]
    field = model.init_field(len(idx), dev)
    lp = lv = 0.0
    w = mk[:, :, None]
    denom = mk.sum() * 2 * k
    for s in range(k):
        pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
        lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom
        lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
    return lp + 0.1 * lv


def train(model, data, args, dev, log):
    P, V, M, mask = [t.float() for t in sf.to_tensors(data[0], dev)]
    ext = tuple(torch.tensor(a, dtype=torch.float32, device=dev) for a in data[1])
    S = P.shape[0]
    iters = args.epochs * (S // args.batch)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    path = os.path.join(args.out_dir, args.tag + ".pt")
    it = 0
    if os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it = st["it"]
        log(f"resumed from it {it}/{iters}")
    g = torch.Generator(device="cpu").manual_seed(args.seed + it)
    t_start, skipped, it_start = time.time(), 0, it
    log(f"scenes {S} N {P.shape[2]} batch {args.batch} iters {iters} ({args.epochs} epochs)")
    while it < iters:
        if STOP:
            save(path, model, opt, it, args)
            log(f"SIGTERM: saved it {it}")
            return it, False
        k, lr = schedule(args, it, iters)
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        t0s = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(dev)
        loss = step_loss(model, P, V, M, mask, ext, idx, k, t0s, dev)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} non-finite, skipped ({skipped})")
        it += 1
        if it % args.log_every == 0:
            el = time.time() - t_start
            log(f"it {it} k {k} lr {lr:.2e} loss {loss.item():.6f} t {el:.0f}s eta {el / (it - it_start) * (iters - it):.0f}s")
        if it % args.ckpt_every == 0 or it == iters:
            if all(torch.isfinite(p).all() for p in model.parameters()):
                save(path, model, opt, it, args)
    return it, True


def err_row(m):
    return [round(m["err"][i], 4) for i in (4, 9, 19)]


@torch.no_grad()
def plain_eval(model, args, dev, log, tag):
    res = {}
    for seed in tsf.EVAL_SEEDS:
        data = tsf.make_data(tsf.EXPS["E"], 48, 100, seed, dev, args.dt, args.eps)
        m, _ = tsf.evaluate_model(model, data, dev, args.dt, args.eps)
        res[f"{tag}_{seed}"] = {"err5_10_20": err_row(m), "err100": m["err"][99], "dE100": m["energy_drift_rel_model"][99],
                                "dE100_true": m["energy_drift_rel_true"][99], "dP100": m["mom_drift_model"][99]}
        log(f"{tag} {seed} err@5/10/20 {err_row(m)} err@100 {m['err'][99]:.3f} dE@100 {m['energy_drift_rel_model'][99]:.3f}")
    return res


@torch.no_grad()
def evaluate(model, args, dev, log):
    t0 = model.t0
    t0.ext_grid = t0.ext_poly = None
    res = plain_eval(model, args, dev, log, "plain")
    if args.baseline:
        base = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", args.dt).to(dev)
        sf.load_compat(base, torch.load(args.baseline, map_location=dev, weights_only=False)["model"])
        res.update(plain_eval(base.eval(), args, dev, log, "expE"))
    data, ext, kinds = td.make_scenes(48, 40, args.seed + 1000, dev)
    exts = tuple(torch.tensor(a, dtype=torch.float32, device=dev) for a in ext)
    for name, zero in (("ext", False), ("ext_chan_zeroed", True)):
        t0.set_external(model, phi0=exts[0], g=exts[1], H=exts[2])
        if zero:
            t0.ext_grid = torch.zeros_like(t0.ext_grid)
        m, _ = tsf.evaluate_model(model, data, dev, args.dt, args.eps)
        res[name] = {"err5_10_20": err_row(m), "dE40": m["energy_drift_rel_model"][39], "dE40_true": m["energy_drift_rel_true"][39]}
        log(f"{name} err@5/10/20 {err_row(m)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default="ext=6,extapply,taper=8,bnd,halo=8")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--tag", default="armA")
    ap.add_argument("--out-dir", default="results/tier0_ext")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--scenes", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--baseline", default=None, help="Exp E checkpoint evaluated on the same plain scenes")
    ap.add_argument("--est", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    signal.signal(signal.SIGTERM, lambda *a: STOP.append(1))

    def log(s):
        print(s, flush=True)

    model = build(args, dev)
    if args.init_ckpt:
        sf.load_compat(model, torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
        log(f"warm start from {args.init_ckpt}")
    log(f"params {sum(p.numel() for p in model.parameters())} spec '{model.t0_spec}' extra input channels {model.t0.cfg.n_extra()}")
    t_gen = time.time()
    data = td.make_scenes(args.scenes if not args.est else 256, args.steps, args.seed, dev)
    log(f"generated {len(data[0])} scenes in {time.time() - t_gen:.0f}s kinds {np.bincount(data[2]).tolist()}")
    if args.est:
        P, V, M, mask = [t.float() for t in sf.to_tensors(data[0], dev)]
        ext = tuple(torch.tensor(a, dtype=torch.float32, device=dev) for a in data[1])
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        per = {}
        for k in (args.k_start, (args.k_start + args.k_end) // 2, args.k_end):
            for rep in range(14):
                if rep == 4:
                    torch.cuda.synchronize()
                    t = time.time()
                idx = torch.randint(P.shape[0], (args.batch,), generator=g).to(dev)
                t0s = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(dev)
                loss = step_loss(model, P, V, M, mask, ext, idx, k, t0s, dev)
                opt.zero_grad()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            per[k] = (time.time() - t) / 10
            log(f"k {k}: {per[k]:.3f} s/it, peak mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        ks = sorted(per)
        iters = args.epochs * (args.scenes // args.batch)
        tot = sum(np.interp(schedule(args, i, iters)[0], ks, [per[k] for k in ks]) for i in range(iters))
        log(f"ESTIMATE {iters} iters = {tot / 3600:.2f} h")
        return
    if not args.eval_only:
        it, done = train(model, data, args, dev, log)
        if not done:
            return
    else:
        st = torch.load(os.path.join(args.out_dir, args.tag + ".pt"), map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
    model.eval()
    res = evaluate(model, args, dev, log)
    json.dump({"t0": args.t0, "args": vars(args), "res": res}, open(os.path.join(args.out_dir, args.tag + "_eval.json"), "w"))
    log("DONE")


if __name__ == "__main__":
    main()
