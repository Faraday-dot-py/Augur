"""Long outward-flow rollout (BH collapse scene as scatter_bh.py: clipped Gaussian sigma 8, vfac 0.3, c 10, relativistic kick-drift-kick) of the model under several
inference configs, with per-tick edge statistics: outgoing bodies and axial reversals in the band 28-32, fraction outside the box, r99, max |v|/c.
Optional exact truth (N <= 2e4) for the same statistics. Saves snapshots (subsample) for rendering. Full state checkpoint every --ckpt-every ticks, resume with --resume.
Usage: python scripts/edge_longroll.py --ckpt X.pt --variant ms_kp_pot_v_g128_ghost --configs fixed=default+kfar@0.05 --n 100000 --steps 1000 --out results/scatter_edge/lr_fixed
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

C, DT, EPS = 10.0, 0.1, 0.5


def make_ic(n, seed):
    rng = np.random.default_rng(seed)
    pos = np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0)
    vel = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    vel -= vel.mean(0)
    return pos, vel


def mom_of(v):
    vm = v.norm(dim=-1, keepdim=True).clamp(max=0.99 * C)
    return v * (1 - (vm / C) ** 2).clamp(min=1e-6).rsqrt()


def axial(p0, p1):
    """axial velocity along the max-norm axis of p1, positive outward, from positions at consecutive ticks."""
    ax = p1.abs().argmax(-1, keepdim=True)
    v = (p1 - p0) / DT
    return torch.gather(v, -1, ax)[..., 0] * torch.sign(torch.gather(p1, -1, ax)[..., 0]), p1.abs().amax(-1)


def band_stats(pm, p, pn, lo=28.0, hi=32.0):
    vb, _ = axial(pm, p)
    va, _ = axial(p, pn)
    c = p.abs().amax(-1)
    sel = (c >= lo) & (c < hi) & (vb > 1.0)
    return int(sel.sum()), int((sel & (va < -1.0)).sum())


def tick_stats(p, v):
    c = p.abs().amax(-1)
    r = p.norm(dim=-1)
    return {"frac_out": float((c > 32).float().mean()), "r99": float(torch.quantile(r[:: max(1, len(r) // 100000)], 0.99)), "vmax_c": float(v.norm(dim=-1).max() / C),
            "n_band": int(((c >= 28) & (c < 32)).sum())}


@torch.no_grad()
def roll(model, pos0, vel0, steps, dev, name, outdir, snap_every, nsnap, ckpt_every, resume, truth=False):
    n = pos0.shape[0]
    ck = f"{outdir}/{name}_state.pt"
    if truth:
        pos = torch.tensor(pos0, dtype=torch.float64, device=dev)
        mom = mom_of(torch.tensor(vel0, dtype=torch.float64, device=dev))
    else:
        pos = torch.tensor(pos0, dtype=torch.float32, device=dev)
        mom = mom_of(torch.tensor(vel0, dtype=torch.float32, device=dev))
    M = torch.ones(1, n, device=dev, dtype=pos.dtype)
    mk = torch.ones_like(M)
    t0 = 0
    series = {"out": [], "rev": [], "stats": []}
    snaps = []
    prev = pos.clone()
    f = None if truth else model.init_field(1, dev)[0]
    a = None
    state = {"t": 0}
    sel_idx = torch.randperm(n, generator=torch.Generator().manual_seed(4738))[:nsnap].to(dev)
    if resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        pos, mom, prev, f, a, t0, series, snaps = st["pos"], st["mom"], st["prev"], st["f"], st["a"], st["t"], st["series"], st["snaps"]
        print(f"resumed {name} at tick {t0}", flush=True)

    def save():
        torch.save({"pos": pos, "mom": mom, "prev": prev, "f": f, "a": a, "t": state["t"], "series": series, "snaps": snaps}, ck)

    def on_term(*_):
        save()
        print("SIGTERM: state saved", flush=True)
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)
    h = DT / 4
    ta = time.time()
    if a is None and not truth:
        a, f = model.force(pos[None], sb.speed(mom, C)[None], M, mk, f)
        a = a[0]
    if a is None and truth:
        a = ops.exact_acc(pos, EPS)
    for t in range(t0, steps):
        state["t"] = t
        pm = prev
        prev = pos.clone()
        if truth:
            for _ in range(4):
                mom = mom + 0.5 * h * a
                pos = pos + h * sb.speed(mom, C)
                a = ops.exact_acc(pos, EPS)
                mom = mom + 0.5 * h * a
        else:
            mom = mom + 0.5 * DT * a
            pos = pos + DT * sb.speed(mom, C)
            a, f = model.force(pos[None], sb.speed(mom, C)[None], M, mk, f)
            a = a[0]
            mom = mom + 0.5 * DT * a
        if t >= 1:
            o, r = band_stats(pm, prev, pos)
            series["out"].append(o)
            series["rev"].append(r)
        if t % 10 == 0:
            series["stats"].append({"t": t, **tick_stats(pos, sb.speed(mom, C))})
        if t % snap_every == 0:
            snaps.append((t, pos[sel_idx].float().cpu().numpy()))
        if t % 100 == 0:
            print(f"{name} t {t} frac_out {series['stats'][-1]['frac_out']:.3f} r99 {series['stats'][-1]['r99']:.1f} out/rev(last100) {sum(series['out'][-100:])}/{sum(series['rev'][-100:])} {(time.time() - ta) / (t - t0 + 1):.2f}s/tick", flush=True)
        if ckpt_every and t % ckpt_every == 0 and t > 0:
            save()
    state["t"] = steps
    np.savez(f"{outdir}/{name}.npz", out=np.array(series["out"]), rev=np.array(series["rev"]), snap_t=np.array([s[0] for s in snaps]), snaps=np.stack([s[1] for s in snaps]),
             stats=json.dumps(series["stats"]))
    o, r = sum(series["out"]), sum(series["rev"])
    print(f"RESULT {name} outgoing-in-band samples {o} reversals {r} frac {r / max(o, 1):.4f} final frac_out {series['stats'][-1]['frac_out']:.3f} r99 {series['stats'][-1]['r99']:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--configs", default="base=default")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--snap-every", type=int, default=5)
    ap.add_argument("--nsnap", type=int, default=30000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    pos0, vel0 = make_ic(args.n, args.seed)
    if args.truth:
        roll(None, pos0, vel0, args.steps, dev, "truth", args.out, args.snap_every, args.nsnap, args.ckpt_every, args.resume, truth=True)
    for spec in args.configs.split(","):
        if not spec:
            continue
        name, opts = spec.split("=")
        m = tsf.build(tsf.EXPS["E"], args.variant, DT).to(dev)
        sf.load_compat(m, torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
        m.eval()
        sf.apply_opts(m, opts.replace("+", ",") + ("" if "kfar" not in opts else ",-graph"))
        roll(m, pos0, vel0, args.steps, dev, name, args.out, args.snap_every, args.nsnap, args.ckpt_every, args.resume)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
