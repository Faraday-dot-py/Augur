"""Train tier 0 on windows harvested from large rollouts (scripts/tiered_gen.py harvest): each window is a 64x64 cut of an N-body
rollout with the direct-sum far-field expansion (a0, J per frame) of everything outside it. Truth positions include the far field.
With extapply the far field is applied exactly (g = -a0, H = -J, quadratic term dropped) and the net learns the near field; without it the
net gets the expansion only through the ext input channels and must reproduce the far force itself (tests whether the channels are usable).

Usage: PYTHONPATH=. python scripts/train_tier0_harvest.py --data results/tier0_hv --t0 "ext=6,extapply,taper=8,bnd,halo=8" --init-ckpt E.pt --tag H1 --out-dir results/tier0_ext
"""
import argparse
import glob
import json
import os
import signal
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m
from scripts import train_scatter_field as tsf
from scripts import train_tier0_ext as tte

STOP = tte.STOP


def load_data(path, dev):
    fs = sorted(glob.glob(os.path.join(path, "hv_s*.npz")))
    cols = {k: [] for k in ("pos", "vel", "mask_f", "mass", "a0", "J", "exp_err")}
    for f in fs:
        z = np.load(f)
        for k in cols:
            cols[k].append(z[k])
    d = {k: torch.tensor(np.concatenate(v), device=dev) for k, v in cols.items()}
    return d, len(fs)


def set_ext(model, d, idx, tt):
    a0, J = d["a0"][idx, tt], d["J"][idx, tt]
    H = torch.stack([-J[:, 0, 0], -0.5 * (J[:, 0, 1] + J[:, 1, 0]), -J[:, 1, 1]], -1)
    model.t0.set_external(model, phi0=torch.zeros(len(idx), device=a0.device), g=-a0, H=H)


def rollout_loss(model, d, idx, t0, k, with_ext=True):
    mk = d["mask_f"][idx, t0].float()
    m = d["mass"][idx].float()
    pos, vel = d["pos"][idx, t0], d["vel"][idx, t0]
    field = model.init_field(len(idx), pos.device)
    lp = lv = 0.0
    denom = mk.sum() * 2 * k
    for s in range(k):
        tt = t0 + s
        if with_ext:
            set_ext(model, d, idx, tt)
        else:
            model.t0.ext_grid = model.t0.ext_poly = None
        pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
        w = (mk * d["mask_f"][idx, t0 + s + 1].float())[:, :, None]
        lp = lp + (((pos - d["pos"][idx, t0 + s + 1]) ** 2) * w).sum() / denom
        lv = lv + (((vel - d["vel"][idx, t0 + s + 1]) ** 2) * w).sum() / denom
    return lp + 0.1 * lv


@torch.no_grad()
def val_err(model, d, vidx, k, with_ext):
    out = []
    for i in range(0, len(vidx), 16):
        idx = vidx[i:i + 16]
        out.append(float(rollout_loss(model, d, idx, torch.zeros(len(idx), dtype=torch.long, device=idx.device), k, with_ext)))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--t0", default="ext=6,extapply,taper=8,bnd,halo=8")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--tag", default="H1")
    ap.add_argument("--out-dir", default="results/tier0_ext")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--norm-ext", action="store_true", help="scale ext input channels to unit rms (extgain from the data)")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))
    log = lambda s: print(s, flush=True)
    d, nsh = load_data(a.data, dev)
    S, T = d["pos"].shape[:2]
    log(f"windows {S} frames {T} N {d['pos'].shape[2]} shards {nsh} mean far-expansion err {float(d['exp_err'].mean()):.3f}")
    rms_a0 = float(d["a0"].pow(2).sum(-1).mean().sqrt())
    log(f"rms a0 {rms_a0:.4f} rms J {float(d['J'].pow(2).sum((-1, -2)).mean().sqrt()):.4f}")
    if a.norm_ext:
        a.t0 += f",extgain={1.0 / rms_a0:.3g}"
    ns = argparse.Namespace(t0=a.t0, dt=a.dt)
    model = tte.build(ns, dev)
    if a.init_ckpt:
        sf.load_compat(model, torch.load(a.init_ckpt, map_location=dev, weights_only=False)["model"])
    log(f"spec '{model.t0_spec}' params {sum(p.numel() for p in model.parameters())}")
    perm = torch.randperm(S, generator=torch.Generator().manual_seed(a.seed)).to(dev)
    nv = max(16, int(S * a.val_frac))
    vidx, tidx = perm[:nv], perm[nv:]
    iters = a.epochs * (len(tidx) // a.batch)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    path = os.path.join(a.out_dir, a.tag + ".pt")
    it = 0
    if os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it = st["it"]
        log(f"resumed from it {it}/{iters}")
    g = torch.Generator(device="cpu").manual_seed(a.seed + it)
    t_start, it_start, streak, skipped = time.time(), it, 0, 0
    kmax = min(a.k_end, T - 1)
    while it < iters:
        if STOP:
            tte.save(path, model, opt, it, a)
            log(f"SIGTERM: saved it {it}")
            return
        frac = it / max(1, iters - 1)
        k = int(round(a.k_start + (kmax - a.k_start) * frac))
        lr = a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = tidx[torch.randint(len(tidx), (a.batch,), generator=g).to(dev)]
        t0s = torch.randint(0, T - k, (a.batch,), generator=g).to(dev)
        loss = rollout_loss(model, d, idx, t0s, k)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
            streak = 0
        else:
            opt.zero_grad()
            skipped += 1
            streak += 1
            if streak >= 20 and os.path.exists(path):
                st = torch.load(path, map_location=dev, weights_only=False)
                model.load_state_dict(st["model"])
                opt.load_state_dict(st["opt"])
                log(f"rolled back to {st['it']}")
                streak = 0
        it += 1
        if it % a.log_every == 0:
            el = time.time() - t_start
            log(f"it {it} k {k} lr {lr:.2e} loss {loss.item():.6f} t {el:.0f}s eta {el / (it - it_start) * (iters - it):.0f}s skipped {skipped}")
        if it % a.ckpt_every == 0 or it == iters:
            if all(torch.isfinite(p).all() for p in model.parameters()):
                tte.save(path, model, opt, it, a)
    model.eval()
    res = {}
    with torch.no_grad():
        res["val_loss_with_ext"] = val_err(model, d, vidx, kmax, True)
        if model.t0.cfg.ext:
            t0 = model.t0
            zero_ext = lambda: None
            res["val_loss_ext_channels_zeroed"] = val_err_zeroed(model, d, vidx, kmax)
        res["val_loss_ext_none"] = val_err(model, d, vidx, kmax, False)
    log(json.dumps(res))
    a.t0 = model.t0_spec
    res.update(tte.plain_eval(model, a, dev, log, "plain"))
    if a.baseline:
        base = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", a.dt).to(dev)
        sf.load_compat(base, torch.load(a.baseline, map_location=dev, weights_only=False)["model"])
        res.update(tte.plain_eval(base.eval(), a, dev, log, "expE"))
    json.dump({"t0": model.t0_spec, "args": vars(a), "res": res}, open(os.path.join(a.out_dir, a.tag + "_eval.json"), "w"))
    log("DONE")


@torch.no_grad()
def val_err_zeroed(model, d, vidx, k):
    out = []
    for i in range(0, len(vidx), 16):
        idx = vidx[i:i + 16]
        t0s = torch.zeros(len(idx), dtype=torch.long, device=idx.device)
        mk = d["mask_f"][idx, 0].float()
        m = d["mass"][idx].float()
        pos, vel = d["pos"][idx, 0], d["vel"][idx, 0]
        field = model.init_field(len(idx), pos.device)
        lp = 0.0
        for s in range(k):
            set_ext(model, d, idx, t0s + s)
            model.t0.ext_grid = torch.zeros_like(model.t0.ext_grid)
            pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
            w = (mk * d["mask_f"][idx, s + 1].float())[:, :, None]
            lp = lp + (((pos - d["pos"][idx, s + 1]) ** 2) * w).sum() / (mk.sum() * 2 * k)
        out.append(float(lp))
    return float(np.mean(out))


if __name__ == "__main__":
    main()
