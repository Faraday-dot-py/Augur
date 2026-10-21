"""Fine-tune Exp E on open-domain scenes (bodies cross the grid box in both directions) to fix the grid-edge force artifact.
Loss as train_scatter_field.train (position + 0.1 velocity, k-step unroll) but masked to bodies whose TRUTH position at the target tick is inside the grid box,
with edge weight 1 + 3*[|coord| > 27]. Data: closed (Exp E generator = same-schedule control), open (scatter_edge_data), or closed + random translation.

Usage: python scripts/train_scatter_edge.py --arm open_full --data open --init-ckpt E.pt --out-dir results/scatter_edge --ckpt-dir checkpoints/edge
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_edge_data as sed
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

EDGE_W = 3.0
EDGE_FROM = 27.0


def exact_acc_batch(pos, mk, eps):
    d = pos[:, None, :, :] - pos[:, :, None, :]
    inv = ((d ** 2).sum(-1) + eps ** 2) ** -1.5 * mk[:, None, :]
    inv = inv * (1 - torch.eye(pos.shape[1], device=pos.device, dtype=pos.dtype))
    return (d * inv[..., None]).sum(2) * mk[..., None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--data", choices=["closed", "open"], default="closed")
    ap.add_argument("--translate", type=float, default=0.0, help="max |shift| per axis of a random per-scene translation of the closed scenes (resampled every iteration)")
    ap.add_argument("--train-only", default="", help="comma list of parameter-name prefixes to train; empty = all")
    ap.add_argument("--loss", choices=["traj", "force"], default="traj", help="force: match exact all-pairs acceleration (all bodies incl. outside the box) of cold-field single evaluations at random states, inside-box bodies, edge-weighted")
    ap.add_argument("--opt", default="default")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out-dir", default="results/scatter_edge")
    ap.add_argument("--ckpt-dir", default="checkpoints/edge")
    args = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = tsf.EXPS["E"]
    t_data = time.time()
    if args.data == "open":
        data = sed.make_open_dataset(args.train, args.steps, args.seed, dev)
    else:
        data = tsf.make_data(cfg, args.train, args.steps, args.seed, dev, 0.1, 0.5)
    P, V, M, mask = [t.float() for t in sf.to_tensors(data, dev)]
    print(f"data {args.data} {tuple(P.shape)} in {time.time() - t_data:.0f}s; frac body-ticks |c|>29.5 "
          f"{float((((P.abs().amax(-1) > 29.5).float() * mask[:, None]).sum() / (mask.sum() * P.shape[1]))):.4f}", flush=True)
    model = tsf.build(cfg, args.variant, 0.1).to(dev)
    sf.apply_opts(model, args.opt, "train")
    sf.load_compat(model, torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
    if args.train_only:
        pre = tuple(args.train_only.split(","))
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(pre))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"arm {args.arm}: trainable params {sum(p.numel() for p in params)} of {sum(p.numel() for p in model.parameters())}", flush=True)
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

    def save():
        if all(torch.isfinite(p).all() for p in model.parameters()):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": state["it"]}, ckpt_path)

    def on_term(*_):
        save()
        print("SIGTERM: checkpoint saved", flush=True)
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)
    g = torch.Generator(device="cpu").manual_seed(args.seed + start)
    S, N = P.shape[0], P.shape[2]
    fns = sf.make_step_fns(model, args.batch, N, args.k_end, dev) if "graphtrain" in getattr(model, "opt", ()) and args.loss == "traj" else None
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
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        t0 = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
        Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
        m, mk = M[idx], mask[idx]
        if args.translate > 0:
            sh = (torch.rand(args.batch, 1, 1, 2, generator=g) * 2 - 1).to(dev) * args.translate
            Pw = Pw + sh
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = model.init_field(args.batch, dev)
        lp = lv = 0.0
        if args.loss == "force":
            c = pos.abs().amax(-1)
            ax = exact_acc_batch(pos.double(), mk.double(), 0.5).float()
            a, _ = model.force(pos, vel, m, mk, field[0] if isinstance(field, tuple) else field)
            w = mk * (c < box).float() * (1 + EDGE_W * (c > EDGE_FROM).float())
            err = (a - ax).norm(dim=-1) / (ax.norm(dim=-1) + 1.0)
            loss = (err ** 2 * w).sum() / w.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if torch.isfinite(loss) and torch.isfinite(gn):
                opt.step()
            if it % args.log_every == 0:
                print(f"it {it} force loss {loss.item():.6f} t {time.time() - t_start:.0f}s", flush=True)
                hist.append((it, float(loss)))
                save()
            continue
        for s in range(k):
            pos, vel, m, field, _ = fns[s](pos, vel, m, mk, field) if fns else sf._step_m(model, pos, vel, m, mk, field)
            c = Pw[:, s + 1].abs().amax(-1)
            w = (mk * (c < box).float() * (1 + EDGE_W * (c > EDGE_FROM).float()))[:, :, None]
            den = w.sum().clamp_min(1.0) * 2
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / den
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / den
        loss = lp / k + 0.1 * lv / k
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
            print(f"it {it} k {k} loss {loss.item():.6f} t {time.time() - t_start:.0f}s", flush=True)
            hist.append((it, float(loss)))
            save()
    state["it"] = args.iters
    save()
    json.dump({"arm": args.arm, "args": vars(args), "loss_hist": hist, "skipped": skipped, "seconds": time.time() - t_start},
              open(f"{args.out_dir}/train_{args.arm}.json", "w"))
    print(f"TRAINED {args.arm} {args.iters} it in {time.time() - t_start:.0f}s skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
