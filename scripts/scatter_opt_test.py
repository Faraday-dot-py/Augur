"""Parity of optimization flags vs eager: inference rollouts (max abs diff of positions) and training (loss, grad rel diff).

Usage: PYTHONPATH=. python scripts/scatter_opt_test.py --ckpt <ckpt> --opts kcache,fastio,cl,graph [--train-opts cl,graphtrain]
"""
import argparse
import copy

import torch

from scripts import scatter_field as sf
from scripts import scatter_opt_bench as sb
from scripts import train_scatter_field as tsf


def load(ckpt, dev, opts):
    m = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    else:
        torch.manual_seed(4738)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
    for o in opts:
        sf.apply_opt(m, o)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opts", default="")
    ap.add_argument("--train-opts", default="")
    ap.add_argument("--ns", default="2,100,300,3000")
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    dev = torch.device("cuda")
    ref = load(args.ckpt, dev, [])
    for n in [int(x) for x in args.ns.split(",")]:
        pos, vel, mass, mask = sb.scene(n, 1, dev)
        Pr, _ = sf.rollout(ref, pos, vel, mass, mask, args.steps)
        Pc, _ = sf.rollout(ref, pos, vel, mass, mask, args.steps)
        d = (Pr - Pc).abs()
        print(f"NOISE eager-vs-eager N {n}: max|dpos| t=1 {d[:, 1].max().item():.2e} t=5 {d[:, 5].max().item():.2e} t=10 {d[:, 10].max().item():.2e} t=20 {d[:, 20].max().item():.2e} t=100 {d[:, -1].max().item():.2e}", flush=True)
    for opts in [o for o in args.opts.split(";") if o]:
        cand = load(args.ckpt, dev, opts.split(","))
        for n in [int(x) for x in args.ns.split(",")]:
            pos, vel, mass, mask = sb.scene(n, 1, dev)
            Pr, Vr = sf.rollout(ref, pos, vel, mass, mask, args.steps)
            Pc, Vc = sf.rollout(cand, pos, vel, mass, mask, args.steps)
            d = (Pr - Pc).abs()
            print(f"INFER [{opts}] N {n}: max|dpos| t=1 {d[:, 1].max().item():.2e} t=5 {d[:, 5].max().item():.2e} t=10 {d[:, 10].max().item():.2e} t=20 {d[:, 20].max().item():.2e} t=100 {d[:, -1].max().item():.2e}", flush=True)
    for n in (1000, 3000, 10000):
        pos, vel, mass, mask = sb.scene(n, 1, dev)
        with torch.no_grad():
            a0 = ref.pp_acc(pos, mass, mask)
            cell = load(args.ckpt, dev, ["cell"])
            a1 = cell.pp_acc_cell(pos, mass, mask)
        print(f"PPFORCE N {n}: max|a_eager-a_cell| {(a0 - a1).abs().max().item():.3e}, max|a| {a0.abs().max().item():.3e}", flush=True)
    for opts in [o for o in args.train_opts.split(";") if o]:
        B, n, k = 8, 50, 6
        pos0, vel0, mass, mask = sb.scene(n, B, dev)
        pos0 = pos0 * 0.5
        res = []
        for use in (None, opts.split(",")):
            m = load(args.ckpt, dev, use or [])
            fns = sf.make_step_fns(m, B, n, k, dev) if use and "graphtrain" in use else None
            m.zero_grad()
            pos, vel, f = pos0, vel0, m.init_field(B, dev)
            loss = 0.0
            for s in range(k):
                pos, vel, _, f, _ = (fns[s](pos, vel, mass, mask, f)) if fns else sf._step_m(m, pos, vel, mass, mask, f)
                loss = loss + (pos ** 2).mean() + 0.1 * (vel ** 2).mean()
            loss.backward()
            res.append((loss.item(), [p.grad.clone() for p in m.parameters() if p.grad is not None]))
        rel = max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(res[0][1], res[1][1]))
        print(f"TRAIN [{opts}]: loss {res[0][0]:.6f} vs {res[1][0]:.6f}, max grad rel diff {rel:.3e}, nparams with grad {len(res[0][1])}/{len(res[1][1])}", flush=True)


if __name__ == "__main__":
    main()
