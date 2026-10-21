"""Escape / core-drift study (docs/debugging/scatter-escape-core-drift.md): relativistic BH-collapse rollout (clipped Gaussian sigma 8, vfac 0.3, c 10, kick-drift-kick on
momentum, dt 0.1) of the ghost-closure Exp E model on a single G=128 grid or tiled (strict G=128 copies, halo 16, ghosts + coarse far term, R1b kernel), with optional
exact fp64 truth (N <= 2e4). Modes: bh (N unit masses) and tracer (N masses + massless tracers launched outward at 0.95c from r 22-26, edge_probe.py scene).
Per tick: fraction outside the 64-box / the tiled domain, r99, count r>70, band-28-32 outgoing/reversal counts, total momentum (all / in box), own core centre
(shrinking sphere) and COM of a fixed core set. Per-body snapshots at SNAP_TICKS for the analysis script. Probes (N <= 2e4, bh): model force vs exact fp64 on the model's own
state by class (core / in-box / band / outside), momfix on and off, model-part breakdown on single grids.
Full state checkpoint every --ckpt-every ticks, --resume, SIGTERM saves.
Usage: PYTHONPATH=. python scripts/escape_roll.py --ckpt ghost_lsq.pt --n 20000 --seed 4738 --arms S1,T4,T8 --truth --out results/escape/n2e4_s4738
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
from scripts import scatter_far as sfar
from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf
from scripts.scatter_tiling import TiledScatter

C, DT, EPS = 10.0, 0.1, 0.5
R1B = "file:results/far_kernels/r1b_mlp.pt+cache"
VARIANT = "ms_kp_pot_v_g128_ghost"
SNAP_TICKS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000)
CORE_TICK = 100
ARMS = {
    "S0": dict(kind="single", kern="old"),
    "S1": dict(kind="single", kern=R1B),
    "S1nm": dict(kind="single", kern=R1B, mf=False),
    "S1g": dict(kind="single", kern=R1B, graph=True),
    "T4": dict(kind="tile", nt=4, tile=32, halo=16, kern=R1B),
    "T4nm": dict(kind="tile", nt=4, tile=32, halo=16, kern=R1B, mf=False),
    "T8": dict(kind="tile", nt=8, tile=32, halo=16, kern=R1B),
    "T8nm": dict(kind="tile", nt=8, tile=32, halo=16, kern=R1B, mf=False),
    "T8w": dict(kind="tile", nt=8, tile=64, halo=16, kern=R1B),
    "T16": dict(kind="tile", nt=16, tile=32, halo=16, kern=R1B),
    "S1in": dict(kind="single", kern=R1B, mf="in"),
    "S1un": dict(kind="single", kern=R1B, mf="un"),
    "T4in": dict(kind="tile", nt=4, tile=32, halo=16, kern=R1B, mf="in"),
    "T8in": dict(kind="tile", nt=8, tile=32, halo=16, kern=R1B, mf="in"),
    "T8nf": dict(kind="tile", nt=8, tile=32, halo=16, kern=R1B, far=False),
    "T8old": dict(kind="tile", nt=8, tile=32, halo=16, kern="old"),
}


def make_ic(n, seed, nt):
    rng = np.random.default_rng(seed)
    pos = np.clip(rng.normal(0.0, 8.0, (n, 2)), -29.0, 29.0)
    vel = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    vel -= vel.mean(0)
    mass = np.ones(n)
    if nt:
        rng2 = np.random.default_rng(seed + 1)
        v0 = rng2.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
        v0 -= v0.mean(0)
        vel = v0
        ang = rng2.uniform(0, 2 * np.pi, nt)
        r0 = rng2.uniform(22, 26, nt)
        pos = np.concatenate([pos, np.stack([r0 * np.cos(ang), r0 * np.sin(ang)], 1)])
        vel = np.concatenate([vel, 0.95 * C * np.stack([np.cos(ang), np.sin(ang)], 1)])
        mass = np.concatenate([mass, np.zeros(nt)])
    return pos, vel, mass


def mom_of(v):
    vm = v.norm(dim=-1, keepdim=True).clamp(max=0.99 * C)
    return v * (1 - (vm / C) ** 2).clamp(min=1e-6).rsqrt()


def centre(pos, iters=(30.0, 15.0, 8.0, 5.0, 5.0)):
    c = pos.median(0).values
    for R in iters:
        sel = (pos - c).norm(dim=-1) < R
        if sel.sum() > 20:
            c = pos[sel].mean(0)
    return c


def axial(p0, p1):
    ax = p1.abs().argmax(-1, keepdim=True)
    v = (p1 - p0) / DT
    return torch.gather(v, -1, ax)[..., 0] * torch.sign(torch.gather(p1, -1, ax)[..., 0])


class Arm:
    """force(pos, speed, field) -> (a, field); holds model / tiler."""

    def __init__(self, name, ckpt, dev, adapt_lmax=14):
        spec = ARMS[name]
        self.name, self.spec, self.kind = name, spec, spec["kind"]
        m = tsf.build(tsf.EXPS["E"], VARIANT, DT).to(dev)
        sf.load_compat(m, torch.load(ckpt, map_location=dev, weights_only=False)["model"])
        m.eval()
        sf.apply_opts(m, ("default" if spec.get("graph") else "kcache,fastio,cl,adapt") if self.kind == "single" else "kcache,adapt")
        m.adapt_lmax = adapt_lmax
        if spec["kern"] != "old":
            sfar.install_spec(m, spec["kern"])
        mf = spec.get("mf", True)
        self.mfin = mf == "in"
        self.mfun = mf == "un"
        mf = mf is True
        self.m, self.dev = m, dev
        if self.kind == "single":
            m.momfix = mf
            self.half = m.extent / 2
        else:
            self.T = TiledScatter(m, spec["nt"], spec["tile"], spec["halo"], ghosts=True, field_x=False, far=spec.get("far", True), momfix=mf, skip_outside=True)
            self.half = spec["nt"] * spec["tile"] / 2

    def init_field(self):
        return self.m.init_field(1, self.dev)[0] if self.kind == "single" else self.T.init_field(self.dev)

    def force(self, pos, v, M, f):
        if self.kind == "single":
            a, fn = self.m._force(pos[None], v[None], M[None], torch.ones_like(M)[None], f) if self.mfun else self.m.force(pos[None], v[None], M[None], torch.ones_like(M)[None], f)
            a = a[0]
            if self.mfun:
                _, an, _, _, _ = self.parts(pos, v, M, f)
                ins = (pos.abs().amax(-1) < self.half) & (M > 0)
                w = (M * ins)[:, None]
                a = a - ins[:, None] * (w * an).sum(0, keepdim=True) / w.sum().clamp_min(1e-9)
            f = fn
        else:
            a, f = self.T.force(pos, v, M, f)
        if self.mfin:
            ins = (pos.abs().amax(-1) < self.half)
            w = (M * ins)[:, None]
            a = a - ins[:, None] * (w * a).sum(0, keepdim=True) / w.sum().clamp_min(1e-9)
        return a, f

    def set_momfix(self, on):
        if self.mfun:
            old, self.mfun = self.mfun, on
            return old
        if self.mfin:
            old, self.mfin = self.mfin, on
            return old
        if self.kind == "single":
            old, self.m.momfix = self.m.momfix, on
        else:
            old, self.T.momfix = self.T.momfix, on
        return old

    def parts(self, pos, v, M, f):
        """single grid only: raw (momfix off) force split into kernel-FFT, U-Net potential gradient and pair parts, each gathered at the bodies."""
        m = self.m
        p, vv, MM = pos[None], v[None], M[None]
        mk = torch.ones_like(MM)
        x = m.scatter(p, vv, MM, mk)
        ak = m.kernel_acc(x[:, :1])
        xin = torch.cat([x, ak, f[None] if f.dim() == 3 else f], 1)
        if "cl" in m.opt:
            xin = xin.contiguous(memory_format=torch.channels_last)
        fld = m.net(xin)
        an = m.neg_grad(fld[:, :1], m.npad)
        a_ker, a_net = m.gather(ak, p)[0], m.gather(an, p)[0]
        a_pp = m.pp_acc(p, MM, mk, m.knn)[0]
        rho = x[0, 0]
        return a_ker, a_net, a_pp, float(rho.sum() * m.h ** 2), float(fld[:, :1].abs().max())


def exact_force(pos, M):
    return ops.exact_acc(pos.double(), EPS, M.double())


def cls_stats(da, ae, am, cen, sel):
    """errors of model force am vs exact ae on selection sel (bool): mean-vector bias, rms relative error, radial/tangential parts about cen."""
    if int(sel.sum()) < 2:
        return None
    e, x, y = (am - ae)[sel].double(), ae[sel].double(), am[sel].double()
    rr = cen.double() - da[sel].double()
    rh = rr / rr.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    th = torch.stack([-rh[:, 1], rh[:, 0]], -1)
    return {"n": int(sel.sum()), "bias": e.mean(0).tolist(), "bias_norm": float(e.mean(0).norm()), "exact_mean": x.mean(0).tolist(), "model_mean": y.mean(0).tolist(),
            "rms_rel": float(e.pow(2).sum(-1).mean().sqrt() / x.pow(2).sum(-1).mean().sqrt().clamp_min(1e-12)),
            "rad_in_exact": float((x * rh).sum(-1).mean()), "rad_in_model": float((y * rh).sum(-1).mean()),
            "tan_rms_err": float(((e * th).sum(-1) ** 2).mean().sqrt()), "rad_rms_err": float(((e * rh).sum(-1) ** 2).mean().sqrt())}


@torch.no_grad()
def probe(arm, pos, mom, M, f, t, half):
    v = sb.speed(mom, C)
    ae = exact_force(pos, M).float()
    old = arm.set_momfix(False)
    if arm.kind == "single":
        am = arm.m._force(pos[None], v[None], M[None], torch.ones_like(M)[None], f)[0][0]
    else:
        am, _ = arm.force(pos, v, M, f)
    arm.set_momfix(old)
    cen = centre(pos)
    d = (pos - cen).norm(dim=-1)
    cm = pos.abs().amax(-1)
    mm = M[:, None]
    abar = (mm * am).sum(0) / mm.sum()
    res = {"t": t, "centre": cen.tolist(), "abar_raw": abar.tolist(), "net_raw_model_all": (mm * am).sum(0).tolist(), "net_exact_all": (mm * ae).sum(0).tolist()}
    inb = cm < 32
    res["net_raw_model_inbox"] = (mm * am)[inb].sum(0).tolist()
    res["net_exact_inbox"] = (mm * ae)[inb].sum(0).tolist()
    res["n_inbox"] = int(inb.sum())
    classes = {"core5": d < 5, "core15": d < 15, "inbox_noncore": inb & (d >= 15) & (cm < 28), "band28_32": (cm >= 28) & (cm < 32), "outside32": cm >= 32}
    res["cls"] = {k: cls_stats(pos, ae, am, cen, s) for k, s in classes.items()}
    am_mf = am - abar
    res["cls_momfix"] = {k: cls_stats(pos, ae, am_mf, cen, s) for k, s in classes.items() if k in ("core5", "core15")}
    if arm.kind == "single":
        ak, an, ap, mass_grid, fmax = arm.parts(pos, v, M, f)
        ap_tot = ak + an + ap
        res["parts_check_rel"] = float((ap_tot - am).norm() / am.norm().clamp_min(1e-12))
        res["mass_grid"], res["mass_inbox_count"], res["field_absmax"] = mass_grid, float(M[inb].sum()), fmax
        res["parts"] = {}
        for k, s in classes.items():
            if int(s.sum()) < 2:
                continue
            res["parts"][k] = {"kernel_mean": ak[s].mean(0).tolist(), "unet_mean": an[s].mean(0).tolist(), "pair_mean": ap[s].mean(0).tolist(), "exact_mean": ae[s].mean(0).tolist(),
                               "kernel_rms": float(ak[s].norm(dim=-1).pow(2).mean().sqrt()), "unet_rms": float(an[s].norm(dim=-1).pow(2).mean().sqrt()),
                               "pair_rms": float(ap[s].norm(dim=-1).pow(2).mean().sqrt()), "exact_rms": float(ae[s].norm(dim=-1).pow(2).mean().sqrt())}
    return res


@torch.no_grad()
def run_arm(arm, pos0, vel0, mass0, steps, dev, outdir, args, shared, truth=False, probe_ticks=()):
    name = "truth" if truth else arm.name
    ck = f"{outdir}/{name}_state.pt"
    dt32 = torch.float64 if truth else torch.float32
    pos = torch.tensor(pos0, dtype=dt32, device=dev)
    mom = mom_of(torch.tensor(vel0, dtype=dt32, device=dev))
    M = torch.tensor(mass0, dtype=dt32, device=dev)
    massive = M > 0
    n_mass = int(massive.sum())
    half = 32.0 if truth else arm.half
    tracer_mode = bool((~massive).any())
    f = None if truth else arm.init_field()
    a, prev, t0 = None, pos.clone(), 0
    ser = {"rows": [], "probes": [], "snap_pos": {}, "snap_mom": {}, "vid": []}
    sel_idx = torch.randperm(pos.shape[0], generator=torch.Generator().manual_seed(4738))[:args.nsnap].to(dev)
    state = {"t": 0}
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        pos, mom, prev, f, a, t0, ser = st["pos"], st["mom"], st["prev"], st["f"], st["a"], st["t"], st["ser"]
        print(f"resumed {name} at tick {t0}", flush=True)

    def save():
        torch.save({"pos": pos, "mom": mom, "prev": prev, "f": f, "a": a, "t": state["t"], "ser": ser}, ck)

    def on_term(*_):
        save()
        print("SIGTERM: state saved", flush=True)
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)
    h = DT / 4

    def force(p, m_, ff):
        if truth:
            return ops.exact_acc(p, EPS, M), None
        return arm.force(p, sb.speed(m_, C), M, ff)

    if a is None:
        a, f = force(pos, mom, f)
    core_ids = shared.get("core_ids")
    ta = time.time()
    for t in range(t0, steps):
        state["t"] = t
        if t in SNAP_TICKS:
            ser["snap_pos"][t] = pos.float().cpu().numpy()
            ser["snap_mom"][t] = mom.float().cpu().numpy()
        pm = prev
        prev = pos.clone()
        if truth:
            for _ in range(4):
                mom = mom + 0.5 * h * a
                pos = pos + h * sb.speed(mom, C)
                a, _ = force(pos, mom, None)
                mom = mom + 0.5 * h * a
        else:
            mom = mom + 0.5 * DT * a
            pos = pos + DT * sb.speed(mom, C)
            a, f = force(pos, mom, f)
            mom = mom + 0.5 * DT * a
        tt = t + 1
        track = massive if not tracer_mode else ~massive
        tp, tpm, tpn = prev[track], pm[track], pos[track]
        vb, va = axial(tpm, tp), axial(tp, tpn)
        cc = tp.abs().amax(-1)
        selb = (cc >= 28) & (cc < 32) & (vb > 1.0)
        out_n, rev_n = int(selb.sum()), int((selb & (va < -1.0)).sum())
        ps, ms = pos[massive], mom[massive]
        cm = ps.abs().amax(-1)
        inb = cm < 32
        r = tpn.norm(dim=-1)
        row = {"t": tt, "frac_out32": float((tpn.abs().amax(-1) > 32).float().mean()), "frac_outdom": float((tpn.abs().amax(-1) > half).float().mean()), "r99": float(torch.quantile(r[:: max(1, len(r) // 100000)], 0.99)),
               "n_r70": int((r > 70).sum()), "n_track": int(track.sum()), "out": out_n, "rev": rev_n, "P_all": ms.sum(0).tolist(), "P_inbox": ms[inb].sum(0).tolist(), "n_inbox": int(inb.sum()),
               "vmax_c": float(sb.speed(mom, C).norm(dim=-1).max() / C)}
        if tt % 5 == 0 and tt >= 5:
            cen = centre(ps)
            d = (ps - cen).norm(dim=-1)
            row["centre"] = cen.tolist()
            row["n_core10"] = int((d < 10).sum())
            row["r50"] = float(d.median())
            row["core_P"] = ms[d < 10].sum(0).tolist()
            row["core_vel"] = sb.speed(ms[d < 10], C).mean(0).tolist() if int((d < 10).sum()) else [0.0, 0.0]
            if core_ids is not None:
                ci = core_ids
                row["set_com"] = pos[ci].mean(0).tolist()
                row["set_P"] = mom[ci].sum(0).tolist()
                row["set_inbox"] = float((pos[ci].abs().amax(-1) < 32).float().mean())
        if tt == CORE_TICK and core_ids is None and not tracer_mode:
            cen = centre(ps)
            d = (pos - cen).norm(dim=-1)
            core_ids = d.argsort()[: max(100, n_mass // 10)]
            shared["core_ids"] = core_ids
            np.save(f"{outdir}/core_ids.npy", core_ids.cpu().numpy())
        if tt in probe_ticks and not truth and not tracer_mode and pos.shape[0] <= 25000:
            pr = probe(arm, pos, mom, M, f, tt, half)
            ser["probes"].append(pr)
        ser["rows"].append(row)
        if tt % args.snap_every == 0:
            ser["vid"].append((tt, pos[sel_idx].float().cpu().numpy()))
        if tt % 100 == 0:
            print(f"{name} t {tt} frac_out32 {row['frac_out32']:.3f} outdom {row['frac_outdom']:.3f} r99 {row['r99']:.1f} n_r70 {row['n_r70']} out/rev {out_n}/{rev_n} "
                  f"{(time.time() - ta) / (tt - t0):.2f}s/tick", flush=True)
        if args.ckpt_every and tt % args.ckpt_every == 0 and tt < steps:
            save()
    state["t"] = steps
    ser["snap_pos"][steps] = pos.float().cpu().numpy()
    ser["snap_mom"][steps] = mom.float().cpu().numpy()
    np.savez(f"{outdir}/{name}.npz", snap_t=np.array(sorted(ser["snap_pos"])), snap_pos=np.stack([ser["snap_pos"][k] for k in sorted(ser["snap_pos"])]),
             snap_mom=np.stack([ser["snap_mom"][k] for k in sorted(ser["snap_mom"])]), vid_t=np.array([s[0] for s in ser["vid"]]), vid=np.stack([s[1] for s in ser["vid"]]),
             mass=mass0, half=half)
    json.dump({"rows": ser["rows"], "probes": ser["probes"]}, open(f"{outdir}/{name}.json", "w"))
    if os.path.exists(ck):
        os.remove(ck)
    last = ser["rows"][-1]
    o, rv = sum(r_["out"] for r_ in ser["rows"]), sum(r_["rev"] for r_ in ser["rows"])
    print(f"RESULT {name} band out {o} rev {rv} frac {rv / max(o, 1):.4f} final frac_out32 {last['frac_out32']:.3f} outdom {last['frac_outdom']:.3f} r99 {last['r99']:.1f} "
          f"n_r70 {last['n_r70']}/{last['n_track']} {time.time() - ta:.0f}s", flush=True)
    return core_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--nt", type=int, default=0, help="massless tracers (0 = bh mode)")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--arms", default="S1")
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--probe-ticks", default="")
    ap.add_argument("--snap-every", type=int, default=5)
    ap.add_argument("--nsnap", type=int, default=30000)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    pos0, vel0, mass0 = make_ic(args.n, args.seed, args.nt)
    shared = {}
    cp = f"{args.out}/core_ids.npy"
    if os.path.exists(cp):
        shared["core_ids"] = torch.tensor(np.load(cp), device=dev)
    pt = tuple(int(x) for x in args.probe_ticks.split(",") if x)
    if args.truth and not (args.resume and os.path.exists(f"{args.out}/truth.npz")):
        run_arm(None, pos0, vel0, mass0, args.steps, dev, args.out, args, shared, truth=True)
        if "core_ids" not in shared and os.path.exists(cp):
            shared["core_ids"] = torch.tensor(np.load(cp), device=dev)
    for name in [x for x in args.arms.split(",") if x]:
        if args.resume and os.path.exists(f"{args.out}/{name}.npz"):
            print("skip finished", name, flush=True)
            continue
        arm = Arm(name, args.ckpt, dev)
        run_arm(arm, pos0, vel0, mass0, args.steps, dev, args.out, args, shared, probe_ticks=pt)
        del arm
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
