"""Closed-loop training of the residual corrector head (scripts/tier0_corrector.py) on the frozen matched tier 0 in the tiled hierarchy
(arm OR: exact external field). Each iteration rolls the frozen model plus the corrector forward from a truth state on multi-clump
scenes and scores the model's OWN rollout against truth (position + 0.1 velocity, optional energy-similarity term matching truth E(t)).
The base runs under stop-gradient; the corrector's gradient flows through the Verlet chain in chunks of --tb steps (truncated BPTT).
Selection on held-out validation scenes (seed 4739), never on the eval seeds 9100/9200/9300.

  PYTHONPATH=. python scripts/train_tier0_corrector.py --s1 results/tier0_ext/tier0-matched-60k-iterations.pt --tag corrector-position-loss
--resume restarts from <out-dir>/<tag>.pt (full state: head, optimizer, iteration, best val); SIGTERM saves and exits."""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_tiered as ttr
from scripts.tier0_corrector import DT, EPS, ResidualCorrector, TiledBase, make_acc, score, verlet

STOP = []


def save(path, corr, opt, it, best, args):
    tmp = path + ".tmp"
    torch.save({"corr": corr.state_dict(), "opt": opt.state_dict(), "it": it, "best": best, "args": vars(args)}, tmp)
    os.replace(tmp, path)


def build(args, dev):
    return ResidualCorrector(width=args.width, use_ext=args.use_ext, momfix=not args.no_momfix).to(dev)


def closed_loop_loss(acc, base, corr, d, idx, t0, k, tb, e_w, backward):
    """Rolls k steps from the truth state at t0 (per scene), returns scalar losses. backward(loss) is called per chunk."""
    dev = d["P"].device
    tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
    Pw, Vw = d["P"][idx[:, None], tt].float(), d["V"][idx[:, None], tt].float()
    m, mk = d["M"][idx].float(), d["mask"][idx].float()
    pos, vel = Pw[:, 0], Vw[:, 0]
    st = base.init_state(len(idx))
    a_prev, st = acc(pos, vel, 0, m, mk, st)
    a_prev = a_prev.detach()
    w, denom = mk[:, :, None], mk.sum() * 2 * k
    tot = {"pos": 0.0, "vel": 0.0, "en": 0.0}
    chunk = 0
    for s in range(k):
        pos, vel, a_prev, st = verlet(acc, pos, vel, a_prev, st, s + 1, m, mk)
        lp = (((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom
        lv = (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
        loss = lp + 0.1 * lv
        tot["pos"] += float(lp.detach())
        tot["vel"] += float(lv.detach())
        if e_w > 0:
            Et = sf.energy(Pw[:, s + 1], Vw[:, s + 1], m, mk, EPS)
            le = (((sf.energy(pos, vel, m, mk, EPS) - Et) / Et.abs().clamp_min(1e-6)) ** 2).mean() / k
            loss = loss + e_w * le
            tot["en"] += float(le.detach())
        chunk = chunk + loss
        if (s + 1) % tb == 0 or s == k - 1:
            backward(chunk)
            chunk = 0
            pos, vel, a_prev = pos.detach(), vel.detach(), a_prev.detach()
    return tot


@torch.no_grad()
def val_loss(base, corr, d, k):
    idx = torch.arange(d["P"].shape[0], device=d["P"].device)
    out = {}
    for name, c in (("corr", corr), ("frozen", None)):
        acc = make_acc(base, c)
        tot = closed_loop_loss(acc, base, c, d, idx, torch.zeros_like(idx), k, k, 0.0, lambda l: None)
        out[name] = tot["pos"] + 0.1 * tot["vel"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="results/tier0_corrector")
    ap.add_argument("--scenes", type=int, default=320)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--val-scenes", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-start", type=int, default=12)
    ap.add_argument("--k-end", type=int, default=40)
    ap.add_argument("--tb", type=int, default=8)
    ap.add_argument("--p-start0", type=float, default=0.5)
    ap.add_argument("--e-w", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--use-ext", action="store_true")
    ap.add_argument("--no-momfix", action="store_true")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--val-seed", type=int, default=4739)
    ap.add_argument("--est", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))
    torch.manual_seed(args.seed)
    log = lambda s: print(s, flush=True)
    base = TiledBase(args.s1, dev, chunk=args.chunk)
    corr = build(args, dev)
    opt = torch.optim.Adam(corr.parameters(), lr=args.lr)
    ck = os.path.join(args.out_dir, args.tag + ".pt")
    best_ck = os.path.join(args.out_dir, args.tag + "_best.pt")
    it, best = 0, float("inf")
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        corr.load_state_dict(st["corr"])
        opt.load_state_dict(st["opt"])
        it, best = st["it"], st["best"]
        log(f"resumed it {it} best val {best:.6f}")
    d = ttr.make_scenes(args.scenes, args.steps, args.seed, dev)
    dv = ttr.make_scenes(args.val_scenes, args.k_end, args.val_seed, dev)
    S, T = d["P"].shape[:2]
    log(f"params {sum(p.numel() for p in corr.parameters())} scenes {S} dev {dev}")
    t_start = time.time()
    while it < args.iters:
        if args.est and it == 3:
            log(f"est {(time.time() - t_start) / 3 * args.iters / 3600:.2f} h for {args.iters} its")
            return
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        for pg in opt.param_groups:
            pg["lr"] = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        g = ttr.gen(args, it, 0)
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        t0 = torch.randint(0, T - k, (args.batch,), generator=g).to(dev)
        t0 = torch.where(torch.rand(args.batch, generator=g).to(dev) < args.p_start0, torch.zeros_like(t0), t0)
        opt.zero_grad()

        def backward(loss):
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()

        tot = closed_loop_loss(make_acc(base, corr), base, corr, d, idx, t0, k, args.tb, args.e_w, backward)
        gn = torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        it += 1
        if it % args.log_every == 0:
            log(f"it {it} k {k} pos {tot['pos']:.6f} vel {tot['vel']:.6f} en {tot['en']:.6f} gn {float(gn):.3f} t {time.time() - t_start:.0f}s")
        if it % args.val_every == 0 or it == args.iters:
            v = val_loss(base, corr, dv, args.k_end)
            log(f"val it {it} corrector {v['corr']:.6f} frozen {v['frozen']:.6f} ratio {v['corr'] / v['frozen']:.4f}")
            if v["corr"] < best:
                best = v["corr"]
                save(best_ck, corr, opt, it, best, args)
        if it % args.ckpt_every == 0 or STOP or it == args.iters:
            save(ck, corr, opt, it, best, args)
        if STOP:
            log("SIGTERM: saved")
            return
    log("done")


if __name__ == "__main__":
    main()
