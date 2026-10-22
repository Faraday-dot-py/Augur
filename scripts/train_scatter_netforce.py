"""Core-drift ARM B: fine-tune the U-Net of Exp E with a soft penalty on the net force error of the model's raw (pre-momfix) force.
Penalty per scene = |sum_in m_i (a_raw_i - a_exact_i)| / sum_in m_i |a_exact_i| over bodies inside the grid box, evaluated at every force call of the k-step unroll
(exact = softened all-pairs gravity at the model's own state, so a system whose true net in-box force is nonzero is not penalised; this is similarity to truth, not zero-sum).
Loss = trajectory loss (position + 0.1 velocity, as train_scatter_field.train) + lam * penalty. --lam 0 is the same-schedule control.
Data: Exp E closed scenes + dense relativistic BH-collapse scenes (N 300..3000, clipped Gaussian sigma 4..8, vfac 0.3, c 10), mixed per iteration with prob --dense-p.
Only parameters whose name starts with --train-only (default net.) are trained; momfix stays on in training (stock).

Usage: python scripts/train_scatter_netforce.py --arm B --lam 0.1 --init-ckpt E.pt --out-dir results/netforce --ckpt-dir checkpoints/netforce
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

C = sf.CENTER
EPS = 0.5


def dense_scene(rng):
    n = int(rng.integers(300, 3001))
    sg = rng.uniform(4.0, 8.0)
    pos = np.clip(rng.normal(0.0, sg, (n, 2)), -29.0, 29.0)
    vel = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    vel -= vel.mean(0)
    return pos + C, vel


def make_dense_dataset(num, steps, seed, device):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        pos, vel = dense_scene(rng)
        out.append(gs.rollout_torch(pos, vel, steps, device, dt=0.1, eps=EPS, relativistic=True, c=10.0))
    return out


def exact_acc(pos, m, mk, eps):
    d = pos[:, None, :, :] - pos[:, :, None, :]
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5 * (m * mk)[:, None, :]
    inv = inv * (1 - torch.eye(pos.shape[1], device=pos.device, dtype=pos.dtype))
    return (d * inv[..., None]).sum(2) * mk[..., None]


def net_penalty(log, m, mk, box):
    tot = 0.0
    for pos, araw in log:
        ae = exact_acc(pos.detach().double(), m.double(), mk.double(), EPS).float()
        w = (m * mk * (pos.detach().abs().amax(-1) < box).float())[..., None]
        net = (w * (araw - ae)).sum(1)
        den = (w * ae).norm(dim=-1).sum(1).clamp_min(1e-6)
        tot = tot + ((net ** 2).sum(-1) + 1e-12).sqrt().div(den).mean()
    return tot / len(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--dense-p", type=float, default=0.5)
    ap.add_argument("--train-only", default="net.")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--n-dense", type=int, default=300)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch-dense", type=int, default=6)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-dir", default="results/netforce")
    ap.add_argument("--ckpt-dir", default="checkpoints/netforce")
    args = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = tsf.EXPS["E"]
    t_data = time.time()
    P, V, M, mask = [t.float() for t in sf.to_tensors(tsf.make_data(cfg, args.train, args.steps, args.seed, dev, 0.1, 0.5), dev)]
    Pd, Vd, Md, maskd = [t.float() for t in sf.to_tensors(make_dense_dataset(args.n_dense, args.steps, args.seed + 7, dev), dev)]
    print(f"data closed {tuple(P.shape)} dense {tuple(Pd.shape)} in {time.time() - t_data:.0f}s", flush=True)
    model = tsf.build(cfg, args.variant, 0.1).to(dev)
    sf.apply_opts(model, "cl", "train")
    sf.load_compat(model, torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
    pre = tuple(args.train_only.split(","))
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(pre))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"arm {args.arm} lam {args.lam}: trainable params {sum(p.numel() for p in params)} of {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    ckpt_path = f"{args.ckpt_dir}/{args.arm}.pt"
    start = 0
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["it"]
        print(f"resumed from it {start}", flush=True)
    state = {"it": start}

    def save(done=False):
        if all(torch.isfinite(p).all() for p in model.parameters()):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": args.iters if done else state["it"]}, ckpt_path)

    def on_term(*_):
        save()
        print("SIGTERM: checkpoint saved", flush=True)
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)
    g = torch.Generator(device="cpu").manual_seed(args.seed + start)
    box = model.extent / 2
    t_start = time.time()
    skipped = 0
    hist = []
    for it in range(start, args.iters):
        state["it"] = it
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        dense = bool(torch.rand(1, generator=g) < args.dense_p)
        Ps, Vs, Ms, Mks, B = (Pd, Vd, Md, maskd, args.batch_dense) if dense else (P, V, M, mask, args.batch)
        idx = torch.randint(Ps.shape[0], (B,), generator=g).to(dev)
        t0 = torch.randint(0, args.steps - k + 1, (B,), generator=g).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
        nmax = int(Mks[idx].sum(1).max())
        Pw, Vw = Ps[idx[:, None], tt][:, :, :nmax], Vs[idx[:, None], tt][:, :, :nmax]
        m, mk = Ms[idx][:, :nmax], Mks[idx][:, :nmax]
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = model.init_field(B, dev)
        model.raw_log = []
        lp = lv = 0.0
        w = mk[:, :, None]
        den = mk.sum() * 2
        for s in range(k):
            pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / den
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / den
        pen = net_penalty(model.raw_log, m, mk, box)
        model.raw_log = None
        loss = lp / k + 0.1 * lv / k + args.lam * pen
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            print(f"it {it} non-finite, skipped ({skipped})", flush=True)
        if it % args.log_every == 0:
            print(f"it {it} k {k} {'dense' if dense else 'closed'} nmax {nmax} traj {(lp / k + 0.1 * lv / k).item():.6f} netratio {float(pen):.4f} loss {loss.item():.6f} t {time.time() - t_start:.0f}s", flush=True)
            hist.append((it, bool(dense), float((lp / k + 0.1 * lv / k).item()), float(pen)))
            save()
    state["it"] = args.iters
    save(done=True)
    json.dump({"arm": args.arm, "args": vars(args), "hist": hist, "skipped": skipped, "seconds": time.time() - t_start}, open(f"{args.out_dir}/train_{args.arm}.json", "w"))
    print(f"TRAINED {args.arm} {args.iters} it in {time.time() - t_start:.0f}s skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
