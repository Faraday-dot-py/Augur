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


def resolve(spec, kind="infer"):
    return sf.resolve_opt(spec, kind)


def check_resolve():
    assert resolve("default") == ["kcache", "fastio", "cl", "graph", "auto"]
    assert resolve("default,-graph") == ["kcache", "fastio", "cl", "auto"]
    assert resolve("none") == [] and resolve("") == []
    assert resolve("default", "train") == ["cl", "graphtrain"]
    assert resolve("none,cell") == ["cell"]
    m = sf.ScatterField(pp=2.0)
    for n, mode in ((100, "dense"), (512, "dense"), (1000, "dense"), (49999, "dense"), (50000, "dense")):
        assert m.pp_mode(n) == mode
    sf.apply_opts(m, "default")
    for n, mode in ((100, "dense"), (512, "dense"), (513, "cell"), (49999, "cell"), (50000, "adapt"), (100000, "adapt")):
        assert m.pp_mode(n) == mode, (n, m.pp_mode(n))
    print("RESOLVE ok; auto mode: N<=512 dense, 513..49999 cell (static cap under graph), >=50000 adapt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opts", default="")
    ap.add_argument("--train-opts", default="")
    ap.add_argument("--ns", default="2,100,300,3000")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cell-ns", default="1000,3000,10000,30000")
    args = ap.parse_args()
    dev = torch.device("cuda")
    check_resolve()
    ref = load(args.ckpt, dev, [])
    for n in [int(x) for x in args.ns.split(",")]:
        pos, vel, mass, mask = sb.scene(n, 1, dev)
        Pr, _ = sf.rollout(ref, pos, vel, mass, mask, args.steps)
        Pc, _ = sf.rollout(ref, pos, vel, mass, mask, args.steps)
        d = (Pr - Pc).abs()
        print(f"NOISE eager-vs-eager N {n}: max|dpos| t=1 {d[:, 1].max().item():.2e} t=5 {d[:, 5].max().item():.2e} t=10 {d[:, 10].max().item():.2e} t=20 {d[:, 20].max().item():.2e} t=100 {d[:, -1].max().item():.2e}", flush=True)
    for opts in [o for o in args.opts.split(";") if o]:
        cand = load(args.ckpt, dev, resolve(opts))
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
            a1 = cell.pp_acc_adapt(pos, mass, mask)
            if n == 3000:
                dense = (torch.randn(1, n, 2, device=dev) * 0.5)
                dense[:, 1500:] = torch.randn(1, n - 1500, 2, device=dev) * 6
                dm = torch.ones(1, n, device=dev)
                print(f"PPDENSE core0.5+halo6 N {n}: max|a_eager-a_adapt| {(ref.pp_acc(dense, dm, dm) - cell.pp_acc_adapt(dense, dm, dm)).abs().max().item():.3e}", flush=True)
        print(f"PPFORCE N {n}: max|a_eager-a_cell| {(a0 - a1).abs().max().item():.3e}, max|a| {a0.abs().max().item():.3e}", flush=True)
    for n in [int(x) for x in args.cell_ns.split(",") if x]:
        pos, vel, mass, mask = sb.scene(n, 1, dev)
        dense = (torch.randn(1, n, 2, device=dev) * 0.5)
        dense[:, n // 2:] = torch.randn(1, n - n // 2, 2, device=dev) * 6
        for tag, (p0, v0) in (("scene", (pos, vel)), ("densecore", (dense, vel))):
            with torch.no_grad():
                f0 = ref.pp_acc(p0, mass, mask) if n <= 10000 else None
                dyn = load(args.ckpt, dev, ["cell"])
                a_dyn = dyn.pp_acc_cell(p0, mass, mask)
                gm = load(args.ckpt, dev, resolve("default"))
                fr = ref.force(p0, v0, mass, mask, ref.init_field(1, dev))[0] if n <= 10000 else None
                fg = gm.force(p0, v0, mass, mask, gm.init_field(1, dev))[0]
                fg = gm.force(p0, v0, mass, mask, gm.init_field(1, dev))[0]
                fd = dyn.force(p0, v0, mass, mask, dyn.init_field(1, dev))[0]
                cap = gm._cap_now
                occ = int(gm.cell_max_occ)
                msg = f"CELLGRAPH {tag} N {n}: cap {cap} max_occ {occ} overflow {occ > cap}, max|a_celldyn-a_cellgraph| {(fd - fg).abs().max().item():.3e}"
                if fr is not None:
                    msg += f", max|a_eager-a_cellgraph| {(fr - fg).abs().max().item():.3e}, max|a| {fr.abs().max().item():.3e}"
                print(msg, flush=True)
                del f0, a_dyn
    for opts in [o for o in args.train_opts.split(";") if o]:
        B, n, k = 8, 50, 6
        pos0, vel0, mass, mask = sb.scene(n, B, dev)
        pos0 = pos0 * 0.5
        res = []
        for use in (None, resolve(opts, "train")):
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
