"""Mesh-only scatter-field rollout of the relativistic black-hole scene at up to N = 1e9 on one GPU, with low-precision body state and chunked scatter/gather.

Same scene and model as scatter_bh_timing.py (Gaussian cluster sigma 8 clipped +-29, vfac 0.3, c 10, relativistic kick-drift-kick on momentum, dt 0.1, Exp E weights,
pair term removed) but the body state (pos, momentum) is stored in --pos-dtype / --mom-dtype, and scatter, gather, kick and drift run over body chunks in fp32 so
temporaries stay O(chunk). Initial conditions are generated on the GPU chunk by chunk.

State is kept in the half-kicked form M = p(t) + dt/2 a(t); one tick is: drift with v(M); scatter; grid force; M += dt (a - mean a). This is the same map as
kick-drift-kick with the two half-kicks of adjacent ticks merged. mean a over bodies is obtained from the grid (gather is the transpose of scatter), so no second
gather is needed for the momentum-conservation correction. Diagnostics use v(M) (half a kick ahead of p).

--mode run:   one rollout; --hist/--snaps save per-tick density histograms (results/<tag>_hist.npz) and a fixed 1e6-body subsample (results/<tag>_snaps.npz: pos, steps).
--mode check: fp32-state vs low-precision-state rollouts from identical ICs, per-tick differences (results/<tag>_check.json).

Usage: PYTHONPATH=. python scripts/scatter_bh_1b.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 1000000000 --steps 100 --hist --snaps --tag scatter_bh_1b
"""
import argparse
import json
import math
import os
import signal
import statistics
import sys
import time

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
HB = 1024


def sync():
    torch.cuda.synchronize()


STOP = [False]


def _on_term(signum, frame):
    STOP[0] = True


def save_state(path, sim, pos, mom, out, t, extra=None):
    tmp = path + ".tmp"
    torch.save({"pos": pos, "mom": mom, "field": sim.field, "out": out, "t": t, **(extra or {})}, tmp)
    os.replace(tmp, path)


def chunks(n, chunk):
    for s in range(0, n, chunk):
        yield s, min(n, s + chunk)


def clump_params(n, args, dev):
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed + 2)
    cen = (torch.randn(args.k, 2, device=dev, generator=g) * args.clump_spread).clamp_(-args.clump_max, args.clump_max)
    vb = torch.randn(args.k, 2, device=dev, generator=g) * (args.vfac * math.sqrt(n / (4 * args.field_sigma)))
    return cen, vb


def make_ic(n, chunk, args, pd, md, ms, dev):
    pos = torch.empty(n, 2, device=dev, dtype=pd)
    mom = torch.empty(n, 2, device=dev, dtype=md)
    vscale = args.vfac * math.sqrt(n / (4 * args.sigma))
    c = args.c
    vsum = torch.zeros(2, dtype=torch.float64, device=dev)
    if args.ic == "clumpy":
        cen, vb = clump_params(n, args, dev)
        nclump = args.clump_frac * n / args.k
        vscale_c = args.vfac * math.sqrt(nclump / (4 * args.clump_sigma))
        vscale_f = args.vfac * math.sqrt(n / (4 * args.field_sigma))
    for pas in range(2):
        g = torch.Generator(device=dev)
        g.manual_seed(args.seed)
        vmean = vsum / n
        for s, e in chunks(n, chunk):
            if args.ic == "clumpy":
                m = e - s
                u = torch.rand(m, device=dev, generator=g)
                cid = torch.randint(0, args.k, (m,), device=dev, generator=g)
                isc = (u < args.clump_frac)[:, None]
                z = torch.randn(m, 2, device=dev, generator=g)
                w = torch.randn(m, 2, device=dev, generator=g)
                p = torch.where(isc, cen[cid] + z * args.clump_sigma, z * args.field_sigma)
                v = torch.where(isc, vb[cid] + w * vscale_c, w * vscale_f)
            else:
                p = torch.randn(e - s, 2, device=dev, generator=g) * args.sigma
                v = torch.randn(e - s, 2, device=dev, generator=g) * vscale
            if args.ic_resample:
                p = torch.where(p.abs() > 29.0, torch.randn(e - s, 2, device=dev, generator=g) * args.sigma, p)
            p = p.clamp_(-29.0, 29.0)
            if pas == 0:
                vsum += v.sum(0, dtype=torch.float64)
                continue
            v -= vmean.float()
            vmag = v.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
            pos[s:e] = p.to(pd)
            mom[s:e] = (v * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt() / ms).to(md)
    return pos, mom


class Sim:
    def __init__(self, model, n, chunk, c, dt, pd, md, ms, dev):
        self.model, self.n, self.chunk, self.c, self.dt, self.pd, self.md, self.ms, self.dev = model, n, chunk, c, dt, pd, md, ms, dev
        self.ones = torch.ones(1, min(chunk, n), device=dev)
        self.field = model.init_field(1, dev)[0]

    def grid_force(self, x):
        m = self.model
        ak = m.kernel_acc(x[:, :1])
        inp = torch.cat([x, ak, self.field], 1)
        self.field = m.net(inp)
        acc = m.neg_grad(self.field[:, :1]) + ak
        w = x[0, 0].double() * (m.h ** 2 / m.in_scale)
        abar = (acc[0].double() * w).sum((1, 2)) / self.n
        return acc, abar.float()

    def scatter_pass(self, pos, mom, drift):
        G = self.model.grid
        x = torch.zeros(1, 3, G, G, device=self.dev)
        for s, e in chunks(self.n, self.chunk):
            v = sb.speed(mom[s:e].float() * self.ms, self.c)
            if drift:
                pos[s:e] = (pos[s:e].float() + self.dt * v).to(self.pd)
            x += self.model.scatter(pos[s:e].float()[None], v[None], self.ones[:, : e - s], self.ones[:, : e - s])
        return x

    def tick(self, pos, mom):
        x = self.scatter_pass(pos, mom, True)
        acc, abar = self.grid_force(x)
        for s, e in chunks(self.n, self.chunk):
            a = self.model.gather(acc, pos[s:e].float()[None])[0]
            mom[s:e] = (mom[s:e].float() + self.dt * (a - abar) / self.ms).to(self.md)

    def init_kick(self, pos, mom):
        x = self.scatter_pass(pos, mom, False)
        acc, abar = self.grid_force(x)
        for s, e in chunks(self.n, self.chunk):
            a = self.model.gather(acc, pos[s:e].float()[None])[0]
            mom[s:e] = (mom[s:e].float() + 0.5 * self.dt * (a - abar) / self.ms).to(self.md)
        return acc


@torch.no_grad()
def diagnostics(sim, pos, mom, sub, hist):
    c, ms = sim.c, sim.ms
    half = sim.model.extent / 2
    out_cnt = torch.zeros((), dtype=torch.float64, device=sim.dev)
    vmax = torch.zeros((), device=sim.dev)
    ke = torch.zeros((), dtype=torch.float64, device=sim.dev)
    bad = torch.zeros((), dtype=torch.float64, device=sim.dev)
    pmax = torch.zeros((), device=sim.dev)
    sp = pos[sub].float()
    cen = sp.median(0).values
    res = {}
    if hist:
        hf = torch.zeros(HB * HB + 1, dtype=torch.int64, device=sim.dev)
        hz4 = torch.zeros_like(hf)
        hz1 = torch.zeros_like(hf)
    for s, e in chunks(sim.n, sim.chunk):
        p = pos[s:e].float()
        m = mom[s:e].float() * ms
        out_cnt += (p.abs() > half).any(-1).sum()
        bad += (~torch.isfinite(p).all(-1) | ~torch.isfinite(m).all(-1)).sum()
        p2 = (m ** 2).sum(-1)
        pmax = torch.maximum(pmax, p2.max().sqrt())
        vmax = torch.maximum(vmax, (p2 / (1 + p2 / c ** 2)).max().sqrt())
        ke += (c ** 2 * ((1 + p2 / c ** 2).sqrt() - 1)).sum(dtype=torch.float64)
        if hist:
            for h, ctr, hw in ((hf, None, 32.0), (hz4, cen, 4.0), (hz1, cen, 1.0)):
                q = (p - ctr + hw) if ctr is not None else (p + hw)
                ij = (q * (HB / (2 * hw))).floor().long()
                ok = ((ij >= 0) & (ij < HB)).all(-1)
                flat = torch.where(ok, ij[:, 1] * HB + ij[:, 0], HB * HB)
                h += torch.bincount(flat, minlength=HB * HB + 1)
    r = (sp - cen).norm(dim=-1)
    res.update(frac_outside=float(out_cnt) / sim.n, vmax_over_c=float(vmax) / c, pmax=float(pmax), mean_ke=float(ke) / sim.n, nonfinite=int(bad),
               core_r45=float(r.quantile(0.45)), r90=float(r.quantile(0.9)), centre=[float(cen[0]), float(cen[1])])
    if hist:
        res["hist"] = [h[:-1].view(HB, HB).int() for h in (hf, hz4, hz1)]
    return res


@torch.no_grad()
def lens_hist(sim, pos, win, res):
    cx, cy, hw = (float(v) for v in win)
    ctr = torch.tensor([cx, cy], device=sim.dev)
    h = torch.zeros(res * res + 1, dtype=torch.int64, device=sim.dev)
    for s, e in chunks(sim.n, sim.chunk):
        ij = ((pos[s:e].float() - ctr + hw) * (res / (2 * hw))).floor().long()
        ok = ((ij >= 0) & (ij < res)).all(-1)
        h += torch.bincount(torch.where(ok, ij[:, 1] * res + ij[:, 0], res * res), minlength=res * res + 1)
    return h[:-1].view(res, res).float().cpu().numpy()


def lens_windows(diag, min_half):
    from scipy.ndimage import gaussian_filter1d
    centre = gaussian_filter1d(np.array([d["centre"] for d in diag]), 4, axis=0, mode="nearest")
    radius = gaussian_filter1d(np.array([d["core_r45"] for d in diag]), 4, mode="nearest")
    return np.concatenate([centre, np.maximum(3.5 * radius, min_half)[:, None]], 1).astype(np.float64)


@torch.no_grad()
def grid_dump(sim, pos, mom):
    m = sim.model
    G = m.grid
    saved = sim.field.clone()
    x = sim.scatter_pass(pos, mom, False)
    rho = x[:, :1]
    phi_far = torch.fft.irfft2(torch.fft.rfft2(rho, s=(2 * G, 2 * G)) * m._kernel_fft(rho), s=(2 * G, 2 * G))[..., :G, :G] * m.h ** 2
    acc, _ = sim.grid_force(x)
    phi_net = sim.field[:, :1]
    sim.field = saved
    return torch.stack([rho[0, 0], acc[0].norm(dim=0), phi_far[0, 0], phi_net[0, 0]]).cpu()


@torch.no_grad()
def force_probe(sim, pos, mom, sub, ntgt, eps=0.5):
    m = sim.model
    saved = sim.field.clone()
    x = sim.scatter_pass(pos, mom, False)
    ak = m.kernel_acc(x[:, :1])
    acc, _ = sim.grid_force(x)
    sim.field = saved
    src = pos[sub].float()
    w = sim.n / src.shape[0]
    tg = src[:ntgt]
    a_tot = m.gather(acc, tg[None])[0]
    a_ker = m.gather(ak, tg[None])[0]
    a_ex = torch.zeros_like(tg)
    for s in range(0, tg.shape[0], 500):
        d = src[None] - tg[s:s + 500, None]
        a_ex[s:s + 500] = w * (d * ((d ** 2).sum(-1, keepdim=True) + eps ** 2) ** -1.5).sum(1)
    r = (tg - src.median(0).values).norm(dim=-1)
    edges = [0, 0.5, 1, 2, 4, 8, 16, 32, 1e9]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (r >= lo) & (r < hi)
        if int(k.sum()) < 20:
            continue
        ae, at, ak_ = a_ex[k], a_tot[k], a_ker[k]
        ne, nt, nk = ae.norm(dim=-1), at.norm(dim=-1), ak_.norm(dim=-1)
        rows.append({"r_lo": lo, "r_hi": hi, "count": int(k.sum()), "exact_mean": float(ne.mean()), "model_mean": float(nt.mean()), "kernel_mean": float(nk.mean()),
                     "model_over_exact_median": float((nt / ne).median()), "kernel_over_exact_median": float((nk / ne).median()),
                     "cos_model_exact_median": float(((at * ae).sum(-1) / (nt * ne)).median()), "relerr_model_median": float(((at - ae).norm(dim=-1) / ne).median())})
    return rows


@torch.no_grad()
def rollout(model, args, pd, md, ms, dev, steps, sub, hist, snaps, label, lens=None):
    n = args.n
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sim = Sim(model, n, args.chunk, args.c, args.dt, pd, md, ms, dev)
    pos, mom = make_ic(n, args.chunk, args, pd, md, ms, dev)
    nm = torch.zeros(2, dtype=torch.float64, device=dev)
    for s_, e_ in chunks(n, args.chunk):
        nm += mom[s_:e_].float().sum(0, dtype=torch.float64) * ms
    net_mom_ic = (nm / n).tolist()
    sync()
    t0 = time.perf_counter()
    sim.init_kick(pos, mom)
    sync()
    out = {"label": label, "n": n, "pos_dtype": str(pd), "mom_dtype": str(md), "mom_scale": ms, "chunk": args.chunk,
           "init_force_ms": (time.perf_counter() - t0) * 1e3, "net_mom_ic_per_body": net_mom_ic, "ms_per_tick": [], "diag": []}
    d = diagnostics(sim, pos, mom, sub, hist)
    hists = [d.pop("hist")] if hist else []
    out["diag"].append(d)
    sn = [pos[sub].float()] if snaps else []
    gd = [grid_dump(sim, pos, mom)] if args.grid_dump else []
    if lens is not None:
        lens[0][0] = lens_hist(sim, pos, lens[1][0], args.lens_res)
        lens[0].flush()
    print(f"[{label}] init force {out['init_force_ms']:.0f} ms, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    ck = os.path.join(args.resume_dir, label + ".ckpt.pt") if args.resume_dir else None
    t_start = 0
    if ck and os.path.exists(ck):
        assert not (snaps or args.grid_dump), "resume supports lens, plain and --hist runs only"
        st = torch.load(ck, map_location=dev, weights_only=False)
        pos.copy_(st["pos"])
        mom.copy_(st["mom"])
        sim.field = st["field"]
        out = st["out"]
        t_start = st["t"]
        if hist:
            hists = [[h.to(dev) for h in x] for x in st["hists"]]
        del st
        print(f"[{label}] resumed from {ck} at tick {t_start}", flush=True)
    t_all = time.perf_counter()
    for t in range(t_start, steps):
        sync()
        t0 = time.perf_counter()
        sim.tick(pos, mom)
        sync()
        out["ms_per_tick"].append((time.perf_counter() - t0) * 1e3)
        d = diagnostics(sim, pos, mom, sub, hist)
        if hist:
            hists.append(d.pop("hist"))
        if args.grid_dump:
            gd.append(grid_dump(sim, pos, mom))
        if t + 1 in args.force_probe:
            out.setdefault("force_probe", {})[t + 1] = force_probe(sim, pos, mom, sub, args.probe_targets)
        d["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["diag"].append(d)
        if snaps:
            sn.append(pos[sub].float())
        if lens is not None:
            lens[0][t + 1] = lens_hist(sim, pos, lens[1][t + 1], args.lens_res)
            lens[0].flush()
        if ck and ((t + 1) % args.ckpt_every == 0 and t + 1 < steps or STOP[0]):
            try:
                save_state(ck, sim, pos, mom, out, t + 1, {"hists": [[h.cpu() for h in x] for x in hists]} if hist else None)
                print(f"[{label}] checkpoint at tick {t + 1}", flush=True)
            except Exception as e:
                print(f"[{label}] checkpoint at tick {t + 1} FAILED: {e!r}", flush=True)
        if STOP[0]:
            print(f"[{label}] SIGTERM: stopped at tick {t + 1}", flush=True)
            sys.exit(0)
        if t % 5 == 0 or t == steps - 1:
            print(f"[{label}] tick {t + 1} {out['ms_per_tick'][-1]:.0f} ms out {d['frac_outside']:.3f} vmax/c {d['vmax_over_c']:.4f} pmax {d['pmax']:.3g} "
                  f"core {d['core_r45']:.3f} nonfinite {d['nonfinite']} peak {d['peak_gb']:.1f} GB wall {time.perf_counter() - t_all:.0f} s", flush=True)
    ms_ = out["ms_per_tick"]
    out.update(first_tick_ms=ms_[0], median_ms=statistics.median(ms_), total_tick_s=sum(ms_) / 1e3, wall_loop_s=time.perf_counter() - t_all,
               peak_gb=torch.cuda.max_memory_allocated() / 1e9, ns_per_body_per_tick=statistics.median(ms_) * 1e6 / n)
    if args.grid_dump:
        np.save(f"results/{args.tag}_grids.npy", torch.stack(gd).numpy())
    return out, hists, sn, pos


def selftest(model, args, dev):
    n = 200000
    a0 = argparse.Namespace(**vars(args))
    a0.n = n
    pos, mom = make_ic(n, 70000, a0, torch.float32, torch.float32, 1.0, dev)
    sim = Sim(model, n, 70000, args.c, args.dt, torch.float32, torch.float32, 1.0, dev)
    x = sim.scatter_pass(pos, mom, False)
    acc, abar = sim.grid_force(x)
    a = torch.cat([model.gather(acc, pos[s:e][None])[0] for s, e in chunks(n, 70000)]) - abar
    sim2 = Sim(model, n, 70000, args.c, args.dt, torch.float32, torch.float32, 1.0, dev)
    ref, _ = model.force(pos[None], sb.speed(mom, args.c)[None], torch.ones(1, n, device=dev), torch.ones(1, n, device=dev), sim2.field)
    err = float((a - ref[0]).abs().max() / ref[0].abs().max())
    print(f"selftest: max|a_chunked - model.force| / max|a| = {err:.2e}", flush=True)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["run", "check"], default="run")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--pos-dtype", default="fp16")
    ap.add_argument("--mom-dtype", default="fp16")
    ap.add_argument("--mom-scale", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--ic", choices=["gaussian", "clumpy"], default="gaussian", help="clumpy: k Gaussian clumps (centres N(0, clump-spread) clamped to +-clump-max, cold bulk + internal velocities) plus a broad field component, mirrors nbody_ic.clumpy")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--clump-frac", type=float, default=0.7)
    ap.add_argument("--clump-sigma", type=float, default=1.5)
    ap.add_argument("--clump-spread", type=float, default=10.0)
    ap.add_argument("--clump-max", type=float, default=20.0)
    ap.add_argument("--field-sigma", type=float, default=10.0)
    ap.add_argument("--ic-resample", action="store_true", help="redraw positions beyond +-29 instead of piling them on the clip lines (square-wall control)")
    ap.add_argument("--nsub", type=int, default=1_000_000)
    ap.add_argument("--hist", action="store_true")
    ap.add_argument("--snaps", action="store_true")
    ap.add_argument("--lens-out", default=None, help="two-pass: pass 1 finds the renderer windows, pass 2 saves per-frame window histograms of all bodies to this .npy memmap")
    ap.add_argument("--lens-res", type=int, default=1080)
    ap.add_argument("--lens-min-half", type=float, default=3.0)
    ap.add_argument("--resume-dir", default=None, help="directory for per-rollout full-state checkpoints (pos, momentum, field, diagnostics); an existing checkpoint is resumed. lens and plain runs only")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--done-file", default=None, help="written after a lens run completes (resume wrappers test for it)")
    ap.add_argument("--kpad", choices=["zero", "ext"], default="zero", help="ext: kernel potential taken one cell beyond each grid side (removes the grid-edge force artifact)")
    ap.add_argument("--tag", default="scatter_bh_1b")
    ap.add_argument("--force-probe", type=lambda v: [int(i) for i in v.split(",") if i], default=[], help="ticks at which to compare grid force with exact pairwise force on the subsample")
    ap.add_argument("--grid-dump", action="store_true", help="save per-tick 128x128 maps (density, |a|, far-field potential, net potential) to results/<tag>_grids.npy")
    ap.add_argument("--probe-targets", type=int, default=20000)
    ap.add_argument("--check-variants", default="fp32:fp32,fp16:fp16")
    args = ap.parse_args()
    if args.resume_dir:
        signal.signal(signal.SIGTERM, _on_term)
        os.makedirs(args.resume_dir, exist_ok=True)
    dev = torch.device("cuda")
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", args.dt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    ops.no_pp(model)
    model.kpad = args.kpad
    res = {"gpu": torch.cuda.get_device_name(), "args": vars(args)}
    if args.ic == "clumpy":
        cen, vb = clump_params(args.n, args, dev)
        res["clumpy_centres"] = cen.tolist()
        res["clumpy_bulk_v"] = vb.tolist()
    json_path = f"results/{args.tag}{'_check' if args.mode == 'check' else ''}.json"

    def write():
        json.dump(res, open(json_path, "w"), indent=1)

    res["selftest_rel_err"] = selftest(model, args, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed + 1)
    sub = torch.unique(torch.randint(0, args.n, (int(args.nsub * 1.01) + 100,), device=dev, generator=g))
    sub = sub[torch.randperm(sub.numel(), device=dev, generator=g)[: args.nsub]].sort().values
    try:
        if args.mode == "check":
            runs = {}
            for v in args.check_variants.split(","):
                pdn, mdn = v.split(":")
                o, h, _, pos = rollout(model, args, DT[pdn], DT[mdn], args.mom_scale, dev, args.steps, sub, True, False, v)
                del pos
                runs[v] = (o, h)
                res.setdefault("runs", {})[v] = o
                write()
            base = args.check_variants.split(",")[0]
            for v, (o, h) in runs.items():
                if v == base:
                    continue
                cmp = []
                for t in range(len(h)):
                    a, b = runs[base][1][t][0].double(), h[t][0].double()
                    a128 = a.view(128, 8, 128, 8).sum((1, 3))
                    b128 = b.view(128, 8, 128, 8).sum((1, 3))
                    d0, d1 = runs[base][0]["diag"][t], o["diag"][t]
                    cmp.append({"tick": t, "tv_128": float((a128 - b128).abs().sum() / (2 * args.n)), "tv_1024": float((a - b).abs().sum() / (2 * args.n)),
                                "core_r45": [d0["core_r45"], d1["core_r45"]], "r90": [d0["r90"], d1["r90"]], "mean_ke": [d0["mean_ke"], d1["mean_ke"]],
                                "vmax_over_c": [d0["vmax_over_c"], d1["vmax_over_c"]], "frac_outside": [d0["frac_outside"], d1["frac_outside"]],
                                "pmax": [d0["pmax"], d1["pmax"]]})
                res.setdefault("compare_vs_" + base, {})[v] = cmp
                print(f"compare {v} vs {base}: last tick tv_128 {cmp[-1]['tv_128']:.4f} core {cmp[-1]['core_r45']} ke {cmp[-1]['mean_ke']}", flush=True)
            for o, _ in runs.values():
                for d in o["diag"]:
                    d.pop("hist", None)
        elif args.lens_out:
            win_path = args.lens_out.replace(".npy", "_windows.npy")
            p1_path = f"results/{args.tag}_pass1.json"
            if args.resume_dir and os.path.exists(win_path) and os.path.exists(p1_path):
                o1 = json.load(open(p1_path))
                win = np.load(win_path)
                print("pass 1 already complete, reusing windows", flush=True)
            else:
                o1, _, _, pos = rollout(model, args, DT[args.pos_dtype], DT[args.mom_dtype], args.mom_scale, dev, args.steps, sub, False, False, args.tag + "_pass1")
                del pos
                win = lens_windows(o1["diag"], args.lens_min_half)
                np.save(win_path, win)
                json.dump(o1, open(p1_path, "w"))
            res["pass1"] = o1
            write()
            resume_mm = bool(args.resume_dir) and os.path.exists(args.lens_out)
            mm = np.lib.format.open_memmap(args.lens_out, mode="r+" if resume_mm else "w+", dtype=np.float32, shape=(len(win), args.lens_res, args.lens_res))
            o, _, _, pos = rollout(model, args, DT[args.pos_dtype], DT[args.mom_dtype], args.mom_scale, dev, args.steps, sub, False, False, args.tag + "_pass2", lens=(mm, win))
            res["run"] = o
            res["lens_windows"] = win.tolist()
            res["pass2_vs_pass1_centre_r45"] = [[a["centre"], a["core_r45"], b["centre"], b["core_r45"]] for a, b in zip(o1["diag"], o["diag"])]
            write()
            print(f"n {args.n}: pass2 median {o['median_ms']:.0f} ms peak {o['peak_gb']:.1f} GB", flush=True)
            if args.done_file:
                open(args.done_file, "w").write("done\n")
        else:
            o, hists, sn, pos = rollout(model, args, DT[args.pos_dtype], DT[args.mom_dtype], args.mom_scale, dev, args.steps, sub, args.hist, args.snaps, args.tag)
            res["run"] = o
            write()
            print(f"n {args.n}: first {o['first_tick_ms']:.0f} ms median {o['median_ms']:.0f} ms ticks {o['total_tick_s']:.1f} s loop {o['wall_loop_s']:.1f} s "
                  f"peak {o['peak_gb']:.1f} GB {o['ns_per_body_per_tick']:.2f} ns/body/tick", flush=True)
            if args.hist:
                cen = np.array([d["centre"] for d in o["diag"]], dtype=np.float32)
                np.savez_compressed(f"results/{args.tag}_hist.npz", full=torch.stack([h[0] for h in hists]).cpu().numpy(),
                                    zoom4=torch.stack([h[1] for h in hists]).cpu().numpy(), zoom1=torch.stack([h[2] for h in hists]).cpu().numpy(),
                                    centre=cen, steps=np.arange(len(hists)), n=args.n, box_half=32.0, zoom_half=np.array([4.0, 1.0]))
                print("hist saved", flush=True)
            if args.snaps:
                np.savez(f"results/{args.tag}_snaps.npz", pos=torch.stack(sn).cpu().numpy(), steps=np.arange(len(sn)), sub_idx=sub.cpu().numpy())
                print("snaps saved", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        res.update(oom=True, oom_message=str(e).splitlines()[0], peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        print("OOM", res["oom_message"], flush=True)
    write()


if __name__ == "__main__":
    main()
