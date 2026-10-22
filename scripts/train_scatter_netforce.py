"""Core-drift ARM B: fine-tune the U-Net of Exp E with a soft penalty on the net force error of the model's raw (pre-momfix) force.
Penalty per scene = |sum_in m_i (a_raw_i - a_exact_i)| / sum_in m_i |a_exact_i| over bodies inside the grid box, evaluated at every force call of a k-step unroll
(exact = softened all-pairs gravity at the model's own state, so a system whose true net in-box force is nonzero is not penalised: similarity to truth, not zero-sum).
Two iteration types, mixed with prob --dense-p:
  closed: Exp E scenes (N 10..300), loss = trajectory loss (position + 0.1 velocity, k 4->20 unroll as train_scatter_field.train) + lam * penalty.
  dense:  states from a bank of stock-E relativistic BH-collapse rollouts (N 4000/8000/20000, snapshot ticks 3..150, with the carried field), unrolled k_dense steps;
          loss = lam * penalty + mu * |a_raw - a_stock|^2 / |a_stock|^2 (in-box), a_stock = frozen stock E evaluated at the student's states (keeps everything but the
          net force near stock E; no truth trajectory exists for these densities, the failing regime is N=2e4).
--lam 0 --mu 0 is the same-schedule control (dense iterations then only add the unroll with zero loss, i.e. skip). Only parameters starting with --train-only (default net.) train;
momfix stays on in training (stock).

Usage: python scripts/train_scatter_netforce.py --arm B --lam 0.1 --init-ckpt E.pt --out-dir results/netforce --ckpt-dir checkpoints/netforce
"""
import argparse
import copy
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

EPS = 0.5
C_LIGHT, DT = 10.0, 0.1
BANK_TICKS = (3, 10, 20, 40, 70, 100, 150)
BANK_N = (4000, 8000, 20000)


def exact_acc(pos, m, mk, eps):
    d = pos[:, None, :, :] - pos[:, :, None, :]
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5 * (m * mk)[:, None, :]
    inv = inv * (1 - torch.eye(pos.shape[1], device=pos.device, dtype=pos.dtype))
    return (d * inv[..., None]).sum(2) * mk[..., None]


def exact_acc_chunked(pos, m, mk, eps):
    return torch.cat([exact_acc(pos[i:i + 1], m[i:i + 1], mk[i:i + 1], eps) for i in range(pos.shape[0])])


def inbox_w(pos, m, mk, box):
    return (m * mk * (pos.abs().amax(-1) < box).float())[..., None]


def net_penalty(log, m, mk, box):
    tot = 0.0
    for pos, _, araw in log:
        ae = exact_acc_chunked(pos.detach().double(), m.double(), mk.double(), EPS).float()
        w = inbox_w(pos.detach(), m, mk, box)
        net = (w * (araw - ae)).sum(1)
        den = (w * ae).norm(dim=-1).sum(1).clamp_min(1e-6)
        tot = tot + ((net ** 2).sum(-1) + 1e-12).sqrt().div(den).mean()
    return tot / len(log)


def distill(log, tlog, m, mk, box):
    tot = 0.0
    for (pos, _, a), (_, _, at) in zip(log, tlog):
        w = inbox_w(pos.detach(), m, mk, box)
        num = (w * (a - at.detach()) ** 2).sum((1, 2))
        den = (w * at.detach() ** 2).sum((1, 2)).clamp_min(1e-9)
        tot = tot + (num / den).mean()
    return tot / len(log)


@torch.no_grad()
def make_bank(model, n_scenes, seed, dev):
    """stock-E relativistic BH-collapse rollouts (clipped Gaussian sigma 8, vfac .3, c 10, kick-drift-kick on momentum); snapshots (pos, vel, field, a) at BANK_TICKS."""
    rng = np.random.default_rng(seed)
    bank = {n: [] for n in BANK_N}
    for i in range(n_scenes):
        n = BANK_N[i % len(BANK_N)]
        pos = torch.tensor(np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0), dtype=torch.float32, device=dev)
        vel = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
        vel -= vel.mean(0)
        vel = torch.tensor(vel, dtype=torch.float32, device=dev)
        vm = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * C_LIGHT)
        mom = vel * (1 - (vm / C_LIGHT) ** 2).clamp(min=1e-6).rsqrt()
        M = torch.ones(1, n, device=dev)
        field = model.init_field(1, dev)[0]
        a, field = model.force(pos[None], sb.speed(mom, C_LIGHT)[None], M, M, field)
        for t in range(1, max(BANK_TICKS) + 1):
            mom = mom + 0.5 * DT * a[0]
            pos = pos + DT * sb.speed(mom, C_LIGHT)
            a, field = model.force(pos[None], sb.speed(mom, C_LIGHT)[None], M, M, field)
            mom = mom + 0.5 * DT * a[0]
            if t in BANK_TICKS:
                bank[n].append((pos.clone(), sb.speed(mom, C_LIGHT).clone(), field.clone(), a.clone()))
    return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--dense-p", type=float, default=0.5)
    ap.add_argument("--k-dense", type=int, default=4)
    ap.add_argument("--n-bank", type=int, default=24)
    ap.add_argument("--train-only", default="net.")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch-dense", type=int, default=2)
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
    model = tsf.build(cfg, args.variant, 0.1).to(dev)
    sf.apply_opts(model, "cl", "train")
    sf.load_compat(model, torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    use_dense = args.lam > 0 or args.mu > 0
    bank = make_bank(teacher, args.n_bank, args.seed + 7, dev) if use_dense else {}
    print(f"data closed {tuple(P.shape)} bank {({n: len(v) for n, v in bank.items()})} in {time.time() - t_data:.0f}s", flush=True)
    pre = tuple(args.train_only.split(","))
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(pre))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"arm {args.arm} lam {args.lam} mu {args.mu}: trainable params {sum(p.numel() for p in params)} of {sum(p.numel() for p in model.parameters())}", flush=True)
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
        if dense and not use_dense:
            continue
        model.raw_log = []
        if dense:
            n = BANK_N[int(torch.randint(len(BANK_N), (1,), generator=g))]
            sel = torch.randint(len(bank[n]), (1 if n > 10000 else args.batch_dense,), generator=g).tolist()
            pos = torch.stack([bank[n][i][0] for i in sel])
            vel = torch.stack([bank[n][i][1] for i in sel])
            fld = torch.cat([bank[n][i][2] for i in sel])
            a0 = torch.cat([bank[n][i][3] for i in sel])
            m = torch.ones(len(sel), n, device=dev)
            mk = m
            nmax = n
            st_s = (fld, a0)
            for _ in range(args.k_dense):
                pos, vel, m, st_s, _ = sf._step_m(model, pos, vel, m, mk, st_s)
            teacher.raw_log = []
            tfield = torch.cat([bank[n][i][2] for i in sel])
            with torch.no_grad():
                for (p_, v_, _) in model.raw_log:
                    _, tfield = teacher.force(p_.detach(), v_.detach(), m, mk, tfield)
            tlog, teacher.raw_log = teacher.raw_log, None
            pen = net_penalty(model.raw_log, m, mk, box)
            dis = distill(model.raw_log, tlog, m, mk, box) if args.mu > 0 else torch.zeros((), device=dev)
            trj = torch.zeros((), device=dev)
            loss = args.lam * pen + args.mu * dis
        else:
            idx = torch.randint(P.shape[0], (args.batch,), generator=g).to(dev)
            t0 = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(dev)
            tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
            nmax = int(mask[idx].sum(1).max())
            Pw, Vw = P[idx[:, None], tt][:, :, :nmax], V[idx[:, None], tt][:, :, :nmax]
            m, mk = M[idx][:, :nmax], mask[idx][:, :nmax]
            pos, vel = Pw[:, 0], Vw[:, 0]
            field = model.init_field(args.batch, dev)
            lp = lv = 0.0
            w = mk[:, :, None]
            den = mk.sum() * 2
            for s in range(k):
                pos, vel, m, field, _ = sf._step_m(model, pos, vel, m, mk, field)
                lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / den
                lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / den
            pen = net_penalty(model.raw_log, m, mk, box)
            trj = lp / k + 0.1 * lv / k
            dis = torch.zeros((), device=dev)
            loss = trj + args.lam * pen
        model.raw_log = None
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            print(f"it {it} non-finite, skipped ({skipped})", flush=True)
        if it % args.log_every == 0 or (it < 20 and it % 4 == 0):
            print(f"it {it} k {k} {'dense' if dense else 'closed'} nmax {nmax} traj {float(trj):.6f} netratio {float(pen):.4f} distill {float(dis):.5f} loss {loss.item():.6f} t {time.time() - t_start:.0f}s", flush=True)
            hist.append((it, bool(dense), float(trj), float(pen), float(dis)))
            if it % args.log_every == 0:
                save()
    state["it"] = args.iters
    save(done=True)
    json.dump({"arm": args.arm, "args": vars(args), "hist": hist, "skipped": skipped, "seconds": time.time() - t_start}, open(f"{args.out_dir}/train_{args.arm}.json", "w"))
    print(f"TRAINED {args.arm} {args.iters} it in {time.time() - t_start:.0f}s skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
