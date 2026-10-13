"""D2b: D5 staged recipe (lr 5e-5, warmup, clip, rollback, k 4->20, k_end 20) plus, opt-in, sigmoid forget gate + large-signal growth penalty. --gate 0 --lam 0 is the D5ctrl code path.
See docs/debugging/error-correction-D2b-contraction.md.

  python scripts/scatter_contact_d2b.py train --tag D2b --gate 1 --lam 1 --k-end 20
  python scripts/scatter_contact_d2b.py eval --cands C=checkpoints_contact/C.pt,D2b=checkpoints_contact/D2b.pt --seeds 9400,9500,9600
Differences from scatter_contact_scenes.run_train / train_scatter_field.train: lr 5e-5 with warmup, k ramp 4->20 then +2 every stage to k_end, micro-batch accumulation
(same batch of 16), snapshot rollback instead of skip when a step blows up (Adam moves each weight ~lr per step whatever the clipped grad norm, so a bad accepted step
poisons every later step), full-state checkpoints every save_every iterations.
"""
import argparse
import copy
import json
import math
import os
import time
import types

import numpy as np
import torch

from scripts import scatter_contact_scenes as cs
from scripts import scatter_d2b as d2b
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import train_scatter_field as tsf

OUT = cs.OUT
CKPT = cs.CKPT


def k_at(it, args):
    if args.k_fixed:
        return args.k_fixed
    if it < args.ramp:
        return int(round(args.k_start + (20 - args.k_start) * it / args.ramp))
    if args.k_end <= 20:
        return 20
    levels = list(range(22, args.k_end + 1, 2))
    return levels[min(len(levels) - 1, (it - args.ramp) * len(levels) // max(1, args.iters - args.ramp))]


def lr_at(it, args):
    frac = it / max(1, args.iters - 1)
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac))) * min(1.0, (it + 1) / args.warmup)


def save(path, model, opt, it, ema, nroll):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "ema": ema, "rollbacks": nroll}, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(model, tensors, geom, rad, args, ckpt, log):
    P, V, M, mask = [t.float() for t in tensors]
    S, dev = P.shape[0], P.device
    gp = list(model.fgate.parameters()) if model.gate else []
    gid = {id(p) for p in gp}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in gid], mult=1.0)]
    if gp:
        groups.append(dict(params=gp, mult=args.gate_mult))
    opt = torch.optim.Adam(groups, lr=args.lr)
    pgen = torch.Generator().manual_seed(args.seed + 1)
    start, ema, nroll = 0, None, 0
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, ema, nroll = st["it"], st["ema"], st["rollbacks"]
        log(f"resumed from it {start}")
    g = torch.Generator(device="cpu").manual_seed(args.seed + start)
    snap = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
    bad_run, last_k, t_start = 0, k_at(start, args), time.time()
    for it in range(start, args.iters):
        k = k_at(it, args)
        if k > last_k:
            if last_k // 10 < k // 10:
                torch.save({"model": model.state_dict(), "it": it}, f"{ckpt}.k{last_k}")
            if ema is not None:
                ema *= 1.3 ** ((k - last_k) / 2)
            last_k = k
        for pg in opt.param_groups:
            pg["lr"] = lr_at(it, args) * pg["mult"]
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        t0 = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
        nb = float(mask[idx].sum())
        opt.zero_grad()
        loss_tot, pen_tot, stt = 0.0, 0.0, {}
        for c in range(0, args.batch, args.micro):
            sl, tc = idx[c:c + args.micro], tt[c:c + args.micro]
            model.set_scene(geom[sl], rad[sl])
            Pw, Vw = P[sl[:, None], tc], V[sl[:, None], tc]
            m, mk = M[sl], mask[sl]
            pos, vel, field = Pw[:, 0], Vw[:, 0], model.init_field(len(sl), dev)
            w = mk[:, :, None]
            loss = 0.0
            for s in range(k):
                pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
                loss = loss + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / (nb * 2 * k) + 0.1 * (((vel - Vw[:, s + 1]) ** 2) * w).sum() / (nb * 2 * k)
                loss = loss + (((m - M[sl]) ** 2) * mk).sum() / (nb * k)
            loss.backward()
            loss_tot += float(loss.detach())
            if args.lam > 0:
                pen, stt = d2b.growth_pen(model, pos, vel, m, mk, field, args.ext, args.rho, args.kappa, args.amp_p, args.amp_max, pgen)
                ps = [p for p in model.parameters() if p.requires_grad]
                main_g = [None if p.grad is None else p.grad.clone() for p in ps]
                opt.zero_grad()
                (args.lam * pen).backward()
                pgn = torch.nn.utils.clip_grad_norm_(ps, args.pclip)
                for p, g0 in zip(ps, main_g):
                    if not bool(torch.isfinite(pgn)):
                        p.grad = g0
                    elif g0 is not None:
                        p.grad = g0 if p.grad is None else p.grad + g0
                pen_tot += float(pen.detach())
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        bad = not (math.isfinite(loss_tot) and bool(torch.isfinite(gn))) or loss_tot > args.abs_cap or (ema is not None and loss_tot > args.ratio * ema)
        if bad:
            opt.zero_grad()
            model.load_state_dict(snap[0])
            opt.load_state_dict(copy.deepcopy(snap[1]))
            nroll += 1
            bad_run += 1
            log(f"it {it} k {k} bad step (loss {loss_tot:.4g} ema {ema}), rolled back to snapshot ({nroll} total)")
            if bad_run > args.max_bad:
                log("too many consecutive bad steps, abort")
                break
            continue
        opt.step()
        bad_run = 0
        ema = loss_tot if ema is None else 0.97 * ema + 0.03 * loss_tot
        if it % args.snap_every == 0:
            snap = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
        if it % args.log_every == 0:
            log(f"it {it} k {k} loss {loss_tot:.6f} ema {ema:.4f} gn {float(gn):.3g} lr {lr_at(it, args):.3g} t {time.time() - t_start:.0f}s mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GB rollbacks {nroll}" + (f" pen {pen_tot:.4g} {stt} gate b {float(model.fgate.bias):.3f} wn {float(model.fgate.weight.norm()):.3f}" if args.lam > 0 else ""))
        if it % args.save_every == 0 and all(torch.isfinite(p).all() for p in model.parameters()):
            save(ckpt, model, opt, it + 1, ema, nroll)
            if it % 2000 == 0:
                save(f"{ckpt}.it{it}", model, opt, it + 1, ema, nroll)
    save(ckpt, model, opt, it + 1, ema, nroll)
    log(f"trained to it {it + 1} in {time.time() - t_start:.0f}s, rollbacks {nroll}")


def run_train(args, dev):
    model = cs.build("C", dev, args.init)
    if args.gate:
        model.add_gate(args.bias, args.gmax)
        model.to(dev)
    model.train()
    scenes, data = cs.train_set(args.train, args.steps, dev, ("c_train",))
    tensors = sf.to_tensors(data, dev)
    n = tensors[0].shape[2]
    geom = cs._geom_stack(scenes, dev)
    rad = cs._rad_pad(scenes, n).to(dev)
    print(f"train data {len(data)} scenes, N max {n}, params {sum(p.numel() for p in model.parameters())}", flush=True)
    os.makedirs(CKPT, exist_ok=True)
    train(model, tensors, geom, rad, args, f"{CKPT}/{args.tag}.pt", lambda s: print(s, flush=True))


def probe(model, es, dev, ticks=100, nsc=24):
    rows = []
    with torch.no_grad():
        for si, s in enumerate(es["scenes"][:nsc]):
            n = len(s["mass"])
            pos = torch.tensor(s["pos"], dtype=torch.float32, device=dev)[None]
            vel = torch.tensor(s["vel"], dtype=torch.float32, device=dev)[None]
            mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
            model.set_scene(cs._geom_stack([s], dev), cs._rad_pad([s], n).to(dev))
            field = model.init_field(1, dev)
            tot, fld, P = [], [], [pos[0].cpu().numpy()]
            for _ in range(ticks):
                f0 = field[0] if isinstance(field, tuple) else field
                a, _ = model._force(pos, vel, mass, mask, f0)
                tot.append(float(a.norm(dim=-1).max()))
                fld.append(float(f0.abs().max()))
                pos, vel, _, field, _ = sf._step_m(model, pos, vel, mass, mask, field)
                P.append(pos[0].cpu().numpy())
            err = np.linalg.norm(np.stack(P)[:, :n] - es["Pt"][si][: ticks + 1, :n], axis=-1).mean(1)
            Pn = np.nan_to_num(np.stack(P)[:, :n], nan=1e9)
            off = float((np.abs(Pn[-1]).max(-1) > 32).mean())
            rows.append(dict(tot=tot, fld=fld, err=[float(err[k]) for k in (5, 20, 40, 100)], off100=off))
    return rows


def probe_summary(rows):
    fix = lambda x: np.nan_to_num(np.array(x, dtype=float), nan=np.inf)
    tot = np.stack([fix(r["tot"]) for r in rows])
    fld = np.stack([fix(r["fld"]) for r in rows])
    g = tot[:, 21:25].max(1) / np.maximum(tot[:, 10:21].max(1), 1e-9)
    return dict(n=len(rows), frac_a_gt_200_t21plus=float((tot[:, 21:].max(1) > 200).mean()), median_a_max_t21plus=float(np.median(tot[:, 21:].max(1))), max_a_max=float(tot.max()),
                median_a_max_t0_20=float(np.median(tot[:, :21].max(1))), median_growth_t21_24_over_t10_20=float(np.median(g)), frac_growth_gt3=float((g > 3).mean()),
                fld_median={str(t): float(np.median(fld[:, t])) for t in (10, 30, 60, 99)}, err_5_20_40_100=[float(np.mean([r["err"][i] for r in rows])) for i in range(4)],
                err_5_20_40_100_median=[float(np.median([r["err"][i] for r in rows])) for i in range(4)], offgrid_frac_100=float(np.mean([r["off100"] for r in rows])))


def run_eval(args, dev):
    cs.EVAL_SEEDS = tuple(int(x) for x in args.seeds.split(","))
    for cand in args.cands.split(","):
        tag, path = cand.split("=")
        model = cs.build("C", dev, path)
        for o in args.eval_opt.split(","):
            sf.apply_opt(model, o)
        cs.run_eval(model, f"d2b_{tag}", types.SimpleNamespace(model="C", ckpt=path, fams=args.fams), dev)
        probes = {}
        for seed in cs.EVAL_SEEDS:
            probes[str(seed)] = probe_summary(probe(model, cs.evalset("c_id", seed, 100, dev), dev))
            print(tag, "probe", seed, json.dumps(probes[str(seed)]), flush=True)
        json.dump(probes, open(f"{OUT}/d2b_probe_{tag}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--tag", default="D5")
    ap.add_argument("--init", default="checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--ramp", type=int, default=3500)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--bias", type=float, default=-6.0)
    ap.add_argument("--gmax", type=float, default=1.0)
    ap.add_argument("--gate-mult", type=float, default=20.0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--ext", type=int, default=12)
    ap.add_argument("--pclip", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--amp-p", type=float, default=0.5)
    ap.add_argument("--amp-max", type=float, default=1000.0)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-fixed", type=int, default=0)
    ap.add_argument("--k-end", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--abs-cap", type=float, default=1e4)
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--snap-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--max-bad", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=ss.TRAIN_SEED)
    ap.add_argument("--cands", default="")
    ap.add_argument("--seeds", default="9400,9500,9600")
    ap.add_argument("--fams", default="c_id,c_nonconvex,c_count,box15")
    ap.add_argument("--eval-opt", default="kcache,fastio")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.mode == "eval":
        cs.build = lambda name, dev, ckpt=None: d2b.build(ckpt, dev)
    run_train(args, dev) if args.mode == "train" else run_eval(args, dev)


if __name__ == "__main__":
    main()
