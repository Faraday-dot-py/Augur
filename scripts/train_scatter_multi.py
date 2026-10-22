"""Train MultiScatterField on mixed-interaction scenes (docs/debugging/scatter-nearfield-retrain.md).

Scenes: 6 radial shapes (pow1 pow2 pow3 yuk8 yuk128 lj; p=1.5 and lam=32 held out) x structures (single species, charges, predator-prey),
amplitude 10^U(-1,1), fp64 direct-sum truth generated on the GPU at start (deterministic in --seed). Loss, k-ramp, lr schedule and the
clip-1.0 / skip-nonfinite rules are those of train_scatter_field.py (Exp E). Full-state checkpoint every --ckpt-every iterations (atomic
rename), SIGTERM saves and exits 143, --resume continues bit-for-bit in schedule (batch sampling is seeded by iteration).
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
from scripts import scatter_multi as sm

STOP = {"flag": False}


def on_term(*_):
    STOP["flag"] = True


def make_cases(args):
    cases, labels = [], []
    if args.only:
        for spec in args.only.split(","):
            shape, st = spec.split(":")
            cases += [(shape, st, None)] * args.per_shape
            labels += [f"{shape}:{st}"] * args.per_shape
        return cases, labels
    rng = np.random.default_rng(args.seed)
    for shape in sm.TRAIN_SHAPES:
        al = sm.allowed(shape)
        w = np.array([sm.STRUCT_W[s] for s in al])
        st = rng.choice(al, args.per_shape, p=w / w.sum())
        cases += [(shape, s, None) for s in st]
        labels += [f"{shape}:{s}" for s in st]
    return cases, labels


def build_data(args, dev, log):
    cases, labels = make_cases(args)
    t0 = time.time()
    d = sm.gen_cases(cases, args.seed, dev, args.steps, tag=1)
    d["label"] = [labels[i] for i in d["sid"].tolist()]
    log(f"data: {len(d['label'])}/{len(cases)} scenes accepted in {time.time() - t0:.0f}s, "
        f"gpu mem {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB")
    return d


def save(path, model, opt, it, args, extra):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "args": vars(args), **extra}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--k-start", type=int, default=2)
    ap.add_argument("--k-ramp", type=float, default=0.8, help="fraction of iterations over which k ramps k-start -> k-end")
    ap.add_argument("--huber", type=float, default=2.0, help="position error (units) beyond which the loss is linear")
    ap.add_argument("--k-end", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--per-shape", type=int, default=12000)
    ap.add_argument("--loss-cap", type=float, default=1000.0, help="skip the update when the batch loss exceeds this (rollout blow-up)")
    ap.add_argument("--only", default="", help="comparator mode: shape:structure,... (single-law training)")
    ap.add_argument("--init-ckpt", default="kern/E_frozen.pt")
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--snap-every", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = f"{args.ckpt_dir}/{args.tag}.pt"
    signal.signal(signal.SIGTERM, on_term)

    def log(s):
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

    if os.path.exists(f"{args.ckpt_dir}/{args.tag}.done"):
        log("done marker present, nothing to do")
        return
    torch.manual_seed(args.seed)
    model = sm.build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    it0, elapsed0 = 0, 0.0
    if args.resume and os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0, elapsed0 = st["it"], st.get("elapsed", 0.0)
        log(f"resumed from it {it0} (elapsed {elapsed0:.0f}s)")
    elif args.init_ckpt:
        model.init_from_e(torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
        log(f"warm start from {args.init_ckpt}")
    log(f"params {sum(p.numel() for p in model.parameters())}")
    data = build_data(args, dev, log)
    S = data["P"].shape[0]
    labels = sorted(set(data["label"]))
    lab_id = torch.tensor([labels.index(x) for x in data["label"]], device=dev)
    nsc = data["n"].to(dev)
    acc_sum = torch.zeros(len(labels), device=dev)
    acc_cnt = torch.zeros(len(labels), device=dev)
    t_start, skipped = time.time(), 0
    it = it0
    hist = []
    while it < args.iters:
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * min(1.0, frac / args.k_ramp)))
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        g = torch.Generator(device="cpu").manual_seed(args.seed * 1000003 + it)
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        t0 = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
        Nb = int(nsc[idx].max())
        Pw = data["P"][idx[:, None], tt][:, :, :Nb]
        Vw = data["V"][idx[:, None], tt][:, :, :Nb]
        mk, typ, cond = data["mask"][idx][:, :Nb], data["typ"][idx][:, :Nb], data["cond"][idx]
        model.set_law(cond, typ)
        m = torch.ones_like(mk)
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = model.init_field(args.batch, dev)
        w = mk[:, :, None]
        denom = mk.sum() * 2 * k
        lp = lv = 0.0
        per = torch.zeros(args.batch, device=dev)
        for s in range(k):
            pos, vel, _, field, _ = sf._step_m(model, pos, vel, m, mk, field)
            pos, vel = pos.clamp(-1e3, 1e3), vel.clamp(-1e3, 1e3)
            dd = (pos - Pw[:, s + 1]).norm(dim=-1, keepdim=True)
            hub = torch.where(dd <= args.huber, dd ** 2, args.huber * (2 * dd - args.huber))
            ep = (hub * w).sum((1, 2)) / denom
            lp = lp + ep.sum()
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
            per = per + ep.detach()
        loss = lp + 0.1 * lv
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn) and loss.item() < args.loss_cap:
            opt.step()
            acc_sum.index_add_(0, lab_id[idx], per)
            acc_cnt.index_add_(0, lab_id[idx], torch.ones_like(per))
        else:
            opt.zero_grad()
            skipped += 1
            if skipped % 50 == 1:
                log(f"it {it} loss {loss.item():.3g} non-finite or above cap, step skipped ({skipped} total)")
        it += 1
        if it % args.log_every == 0:
            log(f"it {it} k {k} lr {lr:.2e} loss {loss.item():.6f} {(time.time() - t_start) / (it - it0):.3f}s/it")
        if it % 1000 == 0:
            bd = {labels[j]: round(float(acc_sum[j] / acc_cnt[j].clamp_min(1)), 6) for j in range(len(labels))}
            log("by-law " + json.dumps(bd))
            hist.append({"it": it, "loss_by_law": bd})
            acc_sum.zero_()
            acc_cnt.zero_()
        stop = STOP["flag"]
        if stop or it % args.ckpt_every == 0 or it == args.iters:
            if all(torch.isfinite(p).all() for p in model.parameters()):
                save(path, model, opt, it, args, {"elapsed": elapsed0 + time.time() - t_start})
                if it % args.snap_every == 0:
                    save(f"{path}.it{it}", model, opt, it, args, {"elapsed": elapsed0 + time.time() - t_start})
            else:
                log("non-finite parameters, checkpoint NOT written")
        if stop:
            log(f"SIGTERM: saved at it {it}, exiting 143")
            sys.exit(143)
    open(f"{args.ckpt_dir}/{args.tag}.done", "w").write(json.dumps({"it": it, "elapsed": elapsed0 + time.time() - t_start}))
    log(f"trained to it {it} in {elapsed0 + time.time() - t_start:.0f}s total")


if __name__ == "__main__":
    main()
