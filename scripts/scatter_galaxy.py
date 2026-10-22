"""Disc-galaxy rollout (scripts/galaxy_ic.py) with the stock Exp E scatter-field model, mesh-only (pair term off, as in the 1e9 BH runs), chunked on one GPU, or the exact fp64 softened all-pairs integrator (--mode truth, N <= ~2e5) from the same IC.

Equal-mass bodies, total mass 300 at every N (body mass 300/N), G=1, eps .5, dt .1, relativistic kick-drift-kick on momentum with c=10 (same maps as scatter_bh_1b.py; mesh state is the half-kicked M = p + dt/2 a, diagnostics use p = M - dt/2 (a - mean a)).
Energy: KE = sum m c^2 (gamma - 1) over all bodies; PE = 1/2 sum m phi_i with phi_i the exact softened potential at --ntgt target bodies from --nsrc source bodies (all of them when N <= nsrc), bound = c^2(gamma-1) + phi_i < 0. Same function for mesh and truth.
Output results/galaxy/<tag>/: meta.json, diag.json, hist_A.npy / hist_B.npy (T,R,R) uint16 = round(3000 log1p(count)) of all bodies in windows +-24 / +-8, snaps.npy (T,K,2) fp16, state/ (full-state checkpoint, mesh and truth).

Usage: PYTHONPATH=. python scripts/scatter_galaxy.py --mode mesh --n 1000000000 --ckpt ~/E_ms_kp_pot_v_g128.pt --tag g1b
"""
import argparse
import json
import os
import shutil
import signal
import statistics
import time

import numpy as np
import torch

from scripts import galaxy_ic
from scripts import scatter_bh as sb
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf
from scripts.scatter_bh_1b import DT, Sim, chunks

EPS = 0.5
QS = 3000.0
STOP = []


class GSim(Sim):
    def __init__(self, model, n, chunk, c, dt, pd, md, mass, dev):
        super().__init__(model, n, chunk, c, dt, pd, md, 1.0, dev)
        self.m = mass
        self.mass = torch.full((1, min(chunk, n)), mass, device=dev)
        self.M = mass * n
        self.acc = None
        self.abar = None

    def grid_force(self, x):
        m = self.model
        ak = m.kernel_acc(x[:, :1])
        inp = torch.cat([x, ak, self.field], 1)
        self.field = m.net(inp)
        acc = m.neg_grad(self.field[:, :1]) + ak
        w = x[0, 0].double() * (m.h ** 2 / m.in_scale)
        abar = (acc[0].double() * w).sum((1, 2)) / self.M
        return acc, abar.float()

    def scatter_pass(self, pos, mom, drift):
        G = self.model.grid
        x = torch.zeros(1, 3, G, G, device=self.dev)
        for s, e in chunks(self.n, self.chunk):
            v = sb.speed(mom[s:e].float(), self.c)
            if drift:
                pos[s:e] = (pos[s:e].float() + self.dt * v).to(self.pd)
            x += self.model.scatter(pos[s:e].float()[None], v[None], self.mass[:, : e - s], self.ones[:, : e - s])
        return x

    def tick(self, pos, mom):
        x = self.scatter_pass(pos, mom, True)
        self.acc, self.abar = self.grid_force(x)
        for s, e in chunks(self.n, self.chunk):
            a = self.model.gather(self.acc, pos[s:e].float()[None])[0]
            mom[s:e] = (mom[s:e].float() + self.dt * (a - self.abar)).to(self.md)

    def init_kick(self, pos, mom):
        x = self.scatter_pass(pos, mom, False)
        self.acc, self.abar = self.grid_force(x)
        for s, e in chunks(self.n, self.chunk):
            a = self.model.gather(self.acc, pos[s:e].float()[None])[0]
            mom[s:e] = (mom[s:e].float() + 0.5 * self.dt * (a - self.abar)).to(self.md)

    def p_of(self, pos_c, mom_c):
        a = self.model.gather(self.acc, pos_c.float()[None])[0]
        return mom_c.float() - 0.5 * self.dt * (a - self.abar)


class TGSim(GSim):
    """Tiled mesh: 4x4 tiles of 32 units (halo 16, G=128 copies, domain +-64) + coarse global far term (scatter_tiling.TiledScatter, pair term off).
    Ghost deposits are the slice of one global G=256 deposit; bodies outside +-64 get no force. Gather is from the assembled tile interiors."""

    def __init__(self, model, n, chunk, c, dt, pd, md, mass, dev, nt=4, tile=32.0, halo=16.0, far=True, field_x=True):
        super().__init__(model, n, chunk, c, dt, pd, md, mass, dev)
        from scripts.scatter_tiling import TiledScatter
        self.tt = TiledScatter(model, nt, tile, halo, ghosts=True, field_x=field_x, far=far)
        self.gg = round(nt * tile / model.h)
        model.grid, model.extent = self.gg, nt * tile
        self.field = self.tt.init_field(dev)

    def grid_force(self, x):
        tt, m = self.tt, self.model
        xt = tt.slice_field(x[0])
        old = tt._scope()
        try:
            ak = m.kernel_acc(xt[:, :1])
            if tt.far:
                ak = ak + tt._far_acc(xt[:, :1])
            f = m.net(torch.cat([xt, ak, self.field], 1))
            acc = m.neg_grad(f[:, :1]) + ak
        finally:
            tt._restore(old)
        self.field = tt.slice_field(tt.assemble(f)) if tt.field_x else f
        accg = tt.assemble(acc)[None]
        w = x[0, 0].double() * (m.h ** 2 / m.in_scale)
        abar = (accg[0].double() * w).sum((1, 2)) / self.M
        return accg, abar.float()


@torch.no_grad()
def potential(tg, src, w, eps=EPS):
    out = torch.empty(tg.shape[0], device=tg.device)
    for s in range(0, tg.shape[0], 64):
        d = src[None] - tg[s:s + 64, None]
        r2 = (d ** 2).sum(-1)
        inv = torch.where(r2 == 0, torch.zeros_like(r2), (r2 + eps ** 2).rsqrt())
        out[s:s + 64] = -w * inv.sum(1)
    return out


@torch.no_grad()
def diagnostics(n, mass, c, chunk, get_chunk, get_tgt, sub, tgt, t):
    """get_chunk(s,e) -> (pos, p) float32; get_tgt(idx) -> (pos, p)."""
    ke = torch.zeros((), dtype=torch.float64, device=sub.device)
    lz = torch.zeros((), dtype=torch.float64, device=sub.device)
    psum = torch.zeros(2, dtype=torch.float64, device=sub.device)
    bad = torch.zeros((), dtype=torch.float64, device=sub.device)
    nvc = torch.zeros((), dtype=torch.float64, device=sub.device)
    out_cnt = torch.zeros((), dtype=torch.float64, device=sub.device)
    out64 = torch.zeros((), dtype=torch.float64, device=sub.device)
    vmax = torch.zeros((), device=sub.device)
    for s, e in chunks(n, chunk):
        p, q = get_chunk(s, e)
        bad += (~torch.isfinite(p).all(-1) | ~torch.isfinite(q).all(-1)).sum()
        q2 = (q ** 2).sum(-1)
        g = (1 + q2 / c ** 2).sqrt()
        ke += (c ** 2 * (g - 1)).sum(dtype=torch.float64)
        v2 = q2 / g ** 2
        vmax = torch.maximum(vmax, v2.max().sqrt())
        nvc += (v2 > c ** 2 * (1 + 1e-6)).sum()
        lz += (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum(dtype=torch.float64)
        psum += q.sum(0, dtype=torch.float64)
        out_cnt += (p.abs() > 32).any(-1).sum()
        out64 += (p.abs() > 64).any(-1).sum()
    ke = float(ke) * mass
    tp, tq = get_tgt(tgt)
    sp, _ = get_tgt(sub)
    phi = potential(tp, sp, mass * n / sub.numel())
    tg = (1 + (tq ** 2).sum(-1) / c ** 2).sqrt()
    e_i = c ** 2 * (tg - 1) + phi
    pe = 0.5 * mass * n * float(phi.mean())
    r = tp.norm(dim=-1)
    return dict(tick=t, ke=ke, pe=pe, e=ke + pe, virial=2 * ke / abs(pe), bound=float((e_i < 0).float().mean()), lz=float(lz) * mass,
                pnet=[float(psum[0]) * mass, float(psum[1]) * mass], nonfinite=int(bad), n_v_ge_c=int(nvc), vmax_over_c=float(vmax) / c,
                frac_outside=float(out_cnt) / n, frac_outside64=float(out64) / n, r25=float(r.quantile(0.25)), r50=float(r.quantile(0.5)), r90=float(r.quantile(0.9)))


@torch.no_grad()
def window_hist(pos_chunks, hw, res, dev):
    h = torch.zeros(res * res + 1, dtype=torch.int64, device=dev)
    for p in pos_chunks:
        ij = ((p + hw) * (res / (2 * hw))).floor().long()
        ok = ((ij >= 0) & (ij < res)).all(-1)
        h += torch.bincount(torch.where(ok, ij[:, 1] * res + ij[:, 0], res * res), minlength=res * res + 1)
    return h[:-1].view(res, res)


def quant(h):
    return torch.round(QS * torch.log1p(h.double())).clamp(max=65535).to(torch.int32).cpu().numpy().astype(np.uint16)


def save_state_mesh(out, pos, mom, sim, t, diag, ms, e0):
    tmp = f"{out}/state_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for name, arr in (("pos", pos), ("mom", mom)):
        mm = np.lib.format.open_memmap(f"{tmp}/{name}.npy", mode="w+", dtype=np.dtype(str(arr.dtype).replace("torch.", "")), shape=tuple(arr.shape))
        for s, e in chunks(arr.shape[0], 50_000_000):
            mm[s:e] = arr[s:e].cpu().numpy()
        mm.flush()
        del mm
    torch.save(dict(field=sim.field, acc=sim.acc, abar=sim.abar, t=t, diag=diag, ms=ms, e0=e0), f"{tmp}/extra.pt")
    shutil.rmtree(f"{out}/state", ignore_errors=True)
    os.rename(tmp, f"{out}/state")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mesh", "truth"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--preset", default="g1")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--variant", default="ms_kp_pot_v_g128")
    ap.add_argument("--c", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--state-dtype", default="fp32")
    ap.add_argument("--chunk", type=int, default=50_000_000)
    ap.add_argument("--rec-every", type=int, default=5)
    ap.add_argument("--diag-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--nsnap", type=int, default=200000)
    ap.add_argument("--ntgt", type=int, default=50000)
    ap.add_argument("--nsrc", type=int, default=1_000_000)
    ap.add_argument("--tile", action="store_true", help="tiled mesh (4x4x32, halo 16, far term)")
    ap.add_argument("--kernel", default="old", help="scatter_far kernel spec (old = shipped kmlp)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = f"results/galaxy/{args.tag}"
    os.makedirs(out, exist_ok=True)
    P = dict(galaxy_ic.PRESETS[args.preset])
    n, c, dt = args.n, args.c, args.dt
    mass = galaxy_ic.total_mass(P) / n
    chunk = min(args.chunk, n)
    pd = md = DT[args.state_dtype] if args.mode == "mesh" else torch.float64
    T = args.steps // args.rec_every + 1
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed + 1)
    sub = torch.arange(n, device=dev) if n <= args.nsrc else torch.unique(torch.randint(0, n, (int(args.nsrc * 1.01) + 100,), device=dev, generator=g))[: args.nsrc]
    sub = sub.sort().values
    tgt = sub[torch.randperm(sub.numel(), device=dev, generator=g)[: args.ntgt]].sort().values
    snap_idx = sub[: args.nsnap]
    meta = dict(args=vars(args), P=P, mass=mass, gpu=torch.cuda.get_device_name(), nsub=int(sub.numel()), ntgt=int(tgt.numel()), T=T)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    mode_rw = "r+" if args.resume else "w+"
    hA = np.lib.format.open_memmap(f"{out}/hist_A.npy", mode=mode_rw, dtype=np.uint16, shape=(T, args.res, args.res))
    hB = np.lib.format.open_memmap(f"{out}/hist_B.npy", mode=mode_rw, dtype=np.uint16, shape=(T, args.res, args.res))
    sn = np.lib.format.open_memmap(f"{out}/snaps.npy", mode=mode_rw, dtype=np.float16, shape=(T, snap_idx.numel(), 2))
    signal.signal(signal.SIGTERM, lambda *a: STOP.append(1))
    wall = time.perf_counter()
    diag, ms, e0, t0 = [], [], None, 0

    def record(t, pos_at):
        if t % args.rec_every:
            return
        i = t // args.rec_every
        hA[i] = quant(window_hist((pos_at(s, e) for s, e in chunks(n, chunk)), 24.0, args.res, dev))
        hB[i] = quant(window_hist((pos_at(s, e) for s, e in chunks(n, chunk)), 8.0, args.res, dev))
        sn[i] = pos_at_idx(snap_idx).half().cpu().numpy()
        if i % 10 == 0:
            hA.flush(); hB.flush(); sn.flush()

    def log_diag(d, t):
        nonlocal e0
        if e0 is None:
            e0 = d["e"]
        d["dE"] = d["e"] / e0 - 1
        d["wall"] = time.perf_counter() - wall
        diag.append(d)
        json.dump(diag, open(f"{out}/diag.json", "w"))
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
        return d["nonfinite"]

    if args.mode == "truth":
        pos32, mom32 = galaxy_ic.make_galaxy(n, P, args.seed, chunk, c, dev)
        pos, p = pos32.double(), mom32.double()
        mvec = torch.full((n,), mass, device=dev, dtype=torch.float64)
        h = dt / args.substeps
        pos_at_idx = lambda idx: pos[idx].float()
        a = ops.exact_acc(pos, EPS, mvec)
        if args.resume:
            st = torch.load(f"{out}/state/extra.pt", map_location=dev)
            pos, p, a, t0, diag, e0 = st["pos"], st["p"], st["a"], st["t"], st["diag"], st["e0"]
        gc = lambda s, e: (pos[s:e].float(), p[s:e].float())
        gt = lambda idx: (pos[idx].float(), p[idx].float())
        if not args.resume:
            log_diag(diagnostics(n, mass, c, chunk, gc, gt, sub, tgt, 0), 0)
            record(0, lambda s, e: pos[s:e].float())
        for t in range(t0 + 1, args.steps + 1):
            ts = time.perf_counter()
            for _ in range(args.substeps):
                p = p + 0.5 * h * a
                pos = pos + h * sb.speed(p, c)
                a = ops.exact_acc(pos, EPS, mvec)
                p = p + 0.5 * h * a
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - ts) * 1e3)
            bad = 0
            if t % args.diag_every == 0:
                bad = log_diag(diagnostics(n, mass, c, chunk, gc, gt, sub, tgt, t), t)
            record(t, lambda s, e: pos[s:e].float())
            if t % args.ckpt_every == 0 or STOP or bad:
                os.makedirs(f"{out}/state", exist_ok=True)
                torch.save(dict(pos=pos, p=p, a=a, t=t, diag=diag, e0=e0), f"{out}/state/extra.pt")
                print(f"tick {t} median {statistics.median(ms):.0f} ms", flush=True)
            if STOP or bad:
                print("STOPPED", t, flush=True)
                break
        meta.update(median_ms=statistics.median(ms), wall_s=time.perf_counter() - wall)
        json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
        hA.flush(); hB.flush(); sn.flush()
        return

    model = tsf.build(tsf.EXPS["E"], args.variant, dt).to(dev)
    model.load_state_dict(torch.load(os.path.expanduser(args.ckpt), map_location=dev, weights_only=False)["model"])
    model.eval()
    ops.no_pp(model)
    if args.kernel != "old":
        from scripts import scatter_far
        scatter_far.install_spec(model, args.kernel)
    torch.cuda.reset_peak_memory_stats()
    sim = (TGSim if args.tile else GSim)(model, n, chunk, c, dt, pd, md, mass, dev)
    with torch.no_grad():
        t0f = time.perf_counter()
        if args.resume:
            pos = torch.empty(n, 2, device=dev, dtype=pd)
            mom = torch.empty(n, 2, device=dev, dtype=md)
            for name, arr in (("pos", pos), ("mom", mom)):
                mm = np.load(f"{out}/state/{name}.npy", mmap_mode="r")
                for s, e in chunks(n, 50_000_000):
                    arr[s:e] = torch.from_numpy(np.ascontiguousarray(mm[s:e])).to(dev)
            st = torch.load(f"{out}/state/extra.pt", map_location=dev)
            sim.field, sim.acc, sim.abar, t0, diag, ms, e0 = st["field"], st["acc"], st["abar"], st["t"], st["diag"], st["ms"], st["e0"]
        else:
            pos, mom = galaxy_ic.make_galaxy(n, P, args.seed, chunk, c, dev, pd, md)
            torch.cuda.synchronize()
            print(f"IC generated in {time.perf_counter() - t0f:.1f} s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            sim.init_kick(pos, mom)
        pos_at_idx = lambda idx: pos[idx].float()
        gc = lambda s, e: (pos[s:e].float(), sim.p_of(pos[s:e], mom[s:e]))
        gt = lambda idx: (pos[idx].float(), sim.p_of(pos[idx], mom[idx]))
        if not args.resume:
            log_diag(diagnostics(n, mass, c, chunk, gc, gt, sub, tgt, 0), 0)
            record(0, lambda s, e: pos[s:e].float())
        for t in range(t0 + 1, args.steps + 1):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            sim.tick(pos, mom)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - ts) * 1e3)
            bad = 0
            if t % args.diag_every == 0:
                bad = log_diag(diagnostics(n, mass, c, chunk, gc, gt, sub, tgt, t), t)
            record(t, lambda s, e: pos[s:e].float())
            if t % 10 == 0:
                print(f"tick {t} {ms[-1]:.0f} ms peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB wall {time.perf_counter() - wall:.0f} s", flush=True)
            if t % args.ckpt_every == 0 or STOP or bad:
                hA.flush(); hB.flush(); sn.flush()
                save_state_mesh(out, pos, mom, sim, t, diag, ms, e0)
                print(f"checkpoint at tick {t}", flush=True)
            if STOP or bad:
                print("STOPPED", t, flush=True)
                break
    meta.update(median_ms=statistics.median(ms), loop_s=sum(ms) / 1e3, peak_gb=torch.cuda.max_memory_allocated() / 1e9, wall_s=time.perf_counter() - wall,
                ns_per_body_per_tick=statistics.median(ms) * 1e6 / n, ticks_done=t if args.steps else 0)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    hA.flush(); hB.flush(); sn.flush()


if __name__ == "__main__":
    main()
