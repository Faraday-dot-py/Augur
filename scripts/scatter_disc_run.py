"""Disc (solar-system formation) rollout with either the scatter-field mesh model (--mode mesh, any N) or the exact fp64 softened all-pairs integrator (--mode truth, N <= ~2e5), same IC (scripts/disc_ic.py), same relativistic-momentum kick-drift-kick as scatter_bh_large.py.

Output in results/disc/<tag>/: snaps_pos.npy, snaps_vel.npy (S, K, 2) float32 at every --snap-every ticks for a fixed subsample (star = row 0), star_track.npy, diag.json, hist.npz (star-centred density frames, mesh mode), state_t<tick>.pt full-state checkpoints (mesh mode), meta.json.

Usage: PYTHONPATH=. python scripts/scatter_disc_run.py --mode mesh --n 10000000 --steps 2500 --ckpt <path> --tag disc_10m
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import disc_ic
from scripts import scatter_bh as sb
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

EPS = 0.5


def to_mom(vel, c):
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    return vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()


def hist_frame(pos, star, hw, res):
    ij = ((pos - star + hw) * (res / (2 * hw))).floor().long()
    ok = ((ij >= 0) & (ij < res)).all(-1)
    flat = torch.where(ok, ij[:, 1] * res + ij[:, 0], torch.full_like(ij[:, 0], res * res))
    return torch.bincount(flat, minlength=res * res + 1)[:-1].view(res, res).int().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mesh", "truth"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--preset", default="d1")
    ap.add_argument("--set", default="", help="comma list key=value overrides of the preset, e.g. Q=0.7,mdisc=15")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--substeps", type=int, default=4, help="truth mode")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--nsub", type=int, default=500000)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--hist-every", type=int, default=5)
    ap.add_argument("--hist-hw", type=float, default=16.0)
    ap.add_argument("--hist-res", type=int, default=512)
    ap.add_argument("--resume", default="")
    ap.add_argument("--nonet", action="store_true", help="mesh mode: kernel-prior force only (grid net off), diagnostic arm")
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--kernel", default="old", help="scatter_far kernel spec, e.g. file:results/far_kernels/r1b_mlp.pt+cache")
    ap.add_argument("--opt", default="", help="comma list of scatter_field opts, e.g. kext")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = f"results/disc/{args.tag}"
    os.makedirs(out, exist_ok=True)
    P = dict(disc_ic.PRESETS[args.preset])
    for kv in filter(None, args.set.split(",")):
        k, v = kv.split("=")
        P[k] = type(P[k])(v)
    n, c, dt = args.n, args.c, args.dt
    pos, vel, mass, ncore = disc_ic.make_disc(n, P, args.seed, dev)
    S = args.steps // args.snap_every + 1
    n_reg = n
    n = mass.numel()
    if n_reg <= args.nsub:
        sub = torch.arange(n, device=dev)
        stride = 1
    else:
        stride = n_reg // args.nsub
        sub = torch.cat([torch.arange(0, n_reg, stride, device=dev)[: args.nsub], torch.arange(n_reg, n, device=dev)])
    sub_wt = mass[sub] * torch.where(sub < n_reg, float(stride), 1.0)
    np.save(f"{out}/sub_wt.npy", sub_wt.cpu().numpy())
    np.save(f"{out}/sub_idx.npy", sub.cpu().numpy())
    K = sub.numel()
    meta = dict(args=vars(args), P=P, n=n, n_reg=n_reg, ncore=ncore, mass_particle=float(mass[0]), K=K, S=S, snap_every=args.snap_every, eps=EPS, gpu=torch.cuda.get_device_name())
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    t0 = 0
    sp = np.lib.format.open_memmap(f"{out}/snaps_pos.npy", mode="r+" if args.resume else "w+", dtype=np.float32, shape=(S, K, 2))
    sv = np.lib.format.open_memmap(f"{out}/snaps_vel.npy", mode="r+" if args.resume else "w+", dtype=np.float32, shape=(S, K, 2))
    star_track, diag, frames = [], [], []
    wall = time.perf_counter()

    def record(t, pos, mom):
        if t % args.snap_every == 0:
            i = t // args.snap_every
            sp[i] = pos[sub].float().cpu().numpy()
            sv[i] = sb.speed(mom[sub], c).float().cpu().numpy()
            sp.flush()
            sv.flush()
        ctr = pos[:ncore].mean(0)
        star_track.append(torch.cat([ctr, sb.speed(mom[:ncore], c).mean(0)]).double().cpu().numpy())
        if t % 10 == 0:
            p = pos.float()
            v = sb.speed(mom, c).float()
            fin = torch.isfinite(p).all(-1) & torch.isfinite(v).all(-1)
            rel = p - p[:ncore].mean(0)
            r = rel.norm(dim=-1)
            rd = r[ncore::4]
            d = dict(tick=t, nonfinite=int((~fin).sum()), vmax=float(v.norm(dim=-1).max()), frac_outside=float((p.abs() > 32).any(-1).float().mean()),
                     core_in2=float((r[:ncore:4] < 2.0).float().mean()), core_r50=float(r[:ncore:4].median()), r50=float(rd.median()), r90=float(rd.quantile(0.9)),
                     centre=[float(ctr[0]), float(ctr[1])], wall=time.perf_counter() - wall)
            diag.append(d)
            if t % 50 == 0:
                print(json.dumps(d), flush=True)
            return d["nonfinite"]
        return 0

    def save():
        np.save(f"{out}/star_track.npy", np.array(star_track))
        json.dump(diag, open(f"{out}/diag.json", "w"))
        if frames:
            np.savez_compressed(f"{out}/hist.npz", frames=np.stack(frames), every=args.hist_every, hw=args.hist_hw)

    if args.mode == "truth":
        h = dt / args.substeps
        mom = to_mom(vel, c)
        a = ops.exact_acc(pos, EPS, mass)
        record(0, pos, mom)
        for t in range(1, args.steps + 1):
            for _ in range(args.substeps):
                mom = mom + 0.5 * h * a
                pos = pos + h * sb.speed(mom, c)
                a = ops.exact_acc(pos, EPS, mass)
                mom = mom + 0.5 * h * a
            if record(t, pos, mom):
                print("NONFINITE", t, flush=True)
                break
        save()
        return
    model = tsf.build(tsf.EXPS["E"], args.variant, dt).to(dev)
    sf.load_compat(model, torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    ops.no_pp(model)
    for o in filter(None, args.opt.split(",")):
        sf.apply_opt(model, o)
    if args.kernel != "old":
        from scripts import scatter_far
        scatter_far.install_spec(model, args.kernel)
    model.nonet = args.nonet
    pos, vel, mass = pos.float(), vel.float(), mass.float()
    mom = to_mom(vel, c)
    mask = torch.ones(1, n, device=dev)
    field = model.init_field(1, dev)[0]
    if args.resume:
        st = torch.load(args.resume, map_location=dev)
        pos, mom, field, t0 = st["pos"], st["mom"], st["field"], st["t"]
        diag = json.load(open(f"{out}/diag.json"))
        diag = [d for d in diag if d["tick"] <= t0]
        star_track = list(np.load(f"{out}/star_track.npy")[: t0 + 1])
    torch.cuda.reset_peak_memory_stats()
    ms = []
    with torch.no_grad():
        a, field = model.force(pos[None], sb.speed(mom, c)[None], mass[None], mask, field)
        a = a[0]
        if not args.resume:
            record(0, pos, mom)
            if args.hist_every:
                frames.append(hist_frame(pos, pos[:ncore].mean(0), args.hist_hw, args.hist_res))
        for t in range(t0 + 1, args.steps + 1):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            mom = mom + 0.5 * dt * a
            pos = pos + dt * sb.speed(mom, c)
            a, field = model.force(pos[None], sb.speed(mom, c)[None], mass[None], mask, field)
            a = a[0]
            mom = mom + 0.5 * dt * a
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - ts) * 1e3)
            bad = record(t, pos, mom)
            if args.hist_every and t % args.hist_every == 0:
                frames.append(hist_frame(pos, pos[:ncore].mean(0), args.hist_hw, args.hist_res))
            if t % args.ckpt_every == 0:
                torch.save(dict(pos=pos, mom=mom, field=field, t=t, P=P, ncore=ncore), f"{out}/state_t{t}.pt")
                save()
                print(f"tick {t} median {np.median(ms):.0f} ms peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            if bad or not np.isfinite(star_track[-1]).all():
                print("NONFINITE", t, flush=True)
                break
    meta.update(median_ms=float(np.median(ms)), loop_s=float(sum(ms) / 1e3), peak_gb=torch.cuda.max_memory_allocated() / 1e9, wall_s=time.perf_counter() - wall)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    save()


if __name__ == "__main__":
    main()
