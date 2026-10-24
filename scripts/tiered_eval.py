"""Evaluation harness for the heterogeneous tiered architecture: the 12 scene groups of docs/theory/heterogeneous-tiered.md
("Training data and evaluation scenes") for any model implementing the interface in scripts/tiered_iface.py.

  python scripts/tiered_eval.py --model scatter:checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/tiered/eval_scatterE.json
  python scripts/tiered_eval.py --model oracle --quick          # harness self-check (truth law as the model)

Groups: bias seams range gravity conservation sparsity objects boundaries calibration exposure generalisation timescale.
A group is skipped (status "skipped: <reason>") when the model lacks the law or capability it needs. Seed 4738 first, held-out
seeds 9100/9200/9300 for headline numbers (summary = mean over held-out). Metrics: err@k (mean per-particle position error, sim units),
dE/dp/dL drift relative to the initial scale (and the truth's own drift), seam jump, cost and memory vs N and empty fraction.
Diagnostic grids (docs/debugging/frame-artifact-review-prompt.md layout: rows = truth / model / density difference, columns = steps,
per-frame autoscale) are written to --grid-dir for the unbiased review."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from scripts import tiered_iface as ti
from scripts import tiered_scenes as sc
from scripts import tiered_truth as tt

GROUPS = ["bias", "seams", "range", "gravity", "conservation", "sparsity", "objects", "boundaries", "calibration", "exposure",
          "generalisation", "timescale"]
KS = (1, 5, 10, 20, 50, 100)
HELD = (9100, 9200, 9300)
FRAMES = {}


def fin(x, cap=1e6):
    return torch.nan_to_num(x, nan=cap, posinf=cap, neginf=-cap)


def tolist(o):
    if isinstance(o, dict):
        return {str(k): tolist(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [tolist(v) for v in o]
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return tolist(np.asarray(o.cpu() if isinstance(o, torch.Tensor) else o).tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def grav_batch(name, n, span, seed, B, dev):
    """B large gravity scenes (same IC family and N) as a batch dict compatible with the tier-0 batches."""
    rng = np.random.default_rng(seed)
    P, V, M = zip(*[sc.big_scene(name, n, span, rng, dev) for _ in range(B)])
    z = lambda a: torch.as_tensor(np.stack(a), device=dev, dtype=torch.float64)
    mass = z(M)
    return dict(pos=z(P), vel=z(V), mass=mass, mask=torch.ones_like(mass), phys=tt.Physics(grav=dict(eps=tt.EPS, g=1.0)),
                occ=torch.zeros(B, 3, sc.GC, sc.GC, device=dev), dt=0.1, sub=4, kind="grav", tags=[dict(ic=name)] * B,
                bound=[dict(kind="none")] * B)


def tier0_batch(kind, seed, B, dev, n_max=128, steps_eq=200, **kw):
    rng = np.random.default_rng(seed)
    fn = kw.pop("fn", None)
    scenes = [getattr(sc, fn)(rng, n_max, **kw) if fn else sc.make_scene(kind, rng, n_max) for _ in range(B)]
    keys = {}
    for s in scenes:
        keys.setdefault((tuple(sorted(s["law"])), s["bound"]["kind"], s["dt"], s["sub"]), []).append(s)
    b = sc.build_batch(max(keys.values(), key=len), dev)
    return sc.equilibrate(b, steps_eq)


def wrap(model, span, sparse=True):
    return model if model.reach >= span else ti.Tiled(model, span, sparse=sparse)


def ctx_for(model, batch, ext, t):
    c = dict(ext=ext, t=t, phys=batch["phys"], bound=batch["bound"], h=batch["dt"] / batch["sub"])
    if "occ" in model.caps:
        c["occ"] = batch["occ"]
    if "periodic" in model.caps and batch["phys"].period is not None:
        c["period"] = batch["phys"].period
    return c


@torch.no_grad()
def truth_roll(batch, dt, steps, ext=None):
    h = batch["dt"] / batch["sub"]
    sub = max(1, int(round(dt / h)))
    return tt.rollout(batch["phys"], batch["pos"], batch["vel"], batch["mass"], batch["mask"], steps, dt, sub, ext=ext)


@torch.no_grad()
def model_roll(model, batch, steps, ext=None, extra_ctx=None):
    pos, vel, mass, mask = batch["pos"], batch["vel"], batch["mass"], batch["mask"]
    state = model.init_state(pos.shape[0], pos.device)
    P, V = [pos], [vel]
    t = 0.0
    bad = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    for k in range(steps):
        ctx = ctx_for(model, batch, ext, t)
        ctx.update(extra_ctx or {})
        pos, vel, state = ti.advance(model, pos, vel, mass, mask, state, ctx)
        ok = torch.isfinite(pos).all((1, 2)) & torch.isfinite(vel).all((1, 2))
        bad = torch.where((bad == 0) & ~ok, torch.full_like(bad, k + 1), bad)
        pos, vel = fin(pos), fin(vel)
        P.append(pos), V.append(vel)
        t += model.dt
    return dict(pos=torch.stack(P, 1), vel=torch.stack(V, 1), first_nonfinite=bad)


def err_curve(Pm, Pt, mask):
    return (fin((Pm - Pt).norm(dim=-1)) * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True)


def at(curve, ks=KS):
    K = curve.shape[-1] - 1
    return {str(k): float(curve[..., k].mean()) for k in ks if k <= K}


def mass_bins(mass, mask, nb=3):
    v = mass[mask > 0]
    if len(v) == 0 or float(v.max() / v.min()) < 1.5:
        return None
    return torch.quantile(torch.log(v).float(), torch.linspace(0, 1, nb + 1)[1:-1].to(v.device)).exp().to(v.dtype)


def breakdown(rm, rt, batch, k):
    """Per-channel (pos/vel x/y) rms error at step k and position error by mass tercile."""
    mk, m = batch["mask"], batch["mass"]
    out = {}
    w = mk.sum()
    for nm in ("pos", "vel"):
        d = fin(rm[nm][:, k] - rt[nm][:, k])
        for c, ax in enumerate("xy"):
            out[f"{nm}_{ax}_rms"] = float((((d[..., c] ** 2) * mk).sum() / w).sqrt())
    e = fin((rm["pos"][:, k] - rt["pos"][:, k]).norm(dim=-1))
    edges = mass_bins(m, mk)
    if edges is not None:
        b = torch.bucketize(m, edges)
        out["pos_err_by_mass_tercile"] = [float((e * mk * (b == i)).sum() / (mk * (b == i)).sum().clamp_min(1)) for i in range(3)]
        out["mass_tercile_edges"] = edges.tolist()
    return out


def conserved(traj, batch, ext, dt):
    """Relative drift curves (B,K+1) of energy (law + external), momentum and angular momentum about the COM."""
    ph, m, mk = batch["phys"], batch["mass"], batch["mask"]
    K1 = traj["pos"].shape[1]
    E = torch.stack([ph.energy(traj["pos"][:, k], traj["vel"][:, k], m, mk) + (ext.pe(traj["pos"][:, k], m, mk, k * dt) if ext is not None else 0.0)
                     for k in range(K1)], 1)
    ke0 = 0.5 * (m * mk * (traj["vel"][:, 0] ** 2).sum(-1)).sum(-1)
    scale = (ke0 + (E[:, 0] - ke0).abs()).clamp_min(1e-12)
    mm = (m * mk)[:, None, :, None]
    p = (mm * traj["vel"]).sum(2)
    pscale = (mm[:, 0] * traj["vel"][:, 0].norm(dim=-1, keepdim=True)).sum((1, 2)).clamp_min(1e-12)
    com = (mm * traj["pos"]).sum(2) / mm.sum(2)
    r = traj["pos"] - com[:, :, None]
    L = (mm[..., 0] * (r[..., 0] * traj["vel"][..., 1] - r[..., 1] * traj["vel"][..., 0])).sum(2)
    lscale = (mm[:, 0, :, 0] * r[:, 0].norm(dim=-1) * traj["vel"][:, 0].norm(dim=-1)).sum(1).clamp_min(1e-12)
    return dict(dE=(E - E[:, :1]) / scale[:, None], dp=(p - p[:, :1]).norm(dim=-1) / pscale[:, None],
                dL=(L - L[:, :1]).abs() / lscale[:, None])


def drift_summary(cm, ct, ks=KS):
    out = {}
    for nm in ("dE", "dp", "dL"):
        K = cm[nm].shape[1] - 1
        out[nm] = {str(k): dict(model=float(fin(cm[nm][:, k]).abs().mean()), truth=float(ct[nm][:, k].abs().mean()),
                                model_minus_truth=float((fin(cm[nm][:, k]) - ct[nm][:, k]).mean())) for k in ks if k <= K}
    return out


def compare(model, batch, steps, ks=KS, ext=None, name=None, keep=False, extra_ctx=None):
    t0 = time.time()
    rt = truth_roll(batch, model.dt, steps, ext)
    rm = model_roll(model, batch, steps, ext, extra_ctx)
    mk = batch["mask"]
    ec = err_curve(rm["pos"], rt["pos"], mk)
    ct, cm = conserved(rt, batch, ext, model.dt), conserved(rm, batch, ext, model.dt)
    kk = max(k for k in ks if k <= steps)
    # constant-velocity reference error, to read err@k against
    cvp = rt["pos"][:, :1] + rt["vel"][:, :1] * (torch.arange(steps + 1, device=mk.device, dtype=torch.float64) * model.dt)[None, :, None, None]
    out = dict(err=at(ec, ks), err_constvel=at(err_curve(cvp, rt["pos"], mk), ks), channels=breakdown(rm, rt, batch, kk),
               drift=drift_summary(cm, ct, ks), first_nonfinite=rm["first_nonfinite"].tolist(), secs=round(time.time() - t0, 2), n=int(mk.sum(-1).mean()))
    if name:
        FRAMES[name] = (rt["pos"][0].cpu().numpy(), rm["pos"][0].cpu().numpy(), batch["mass"][0].cpu().numpy(), batch["mask"][0].cpu().numpy())
    return out


def agg(per_seed):
    """Summary over held-out seeds (mean of every numeric leaf) alongside the seed-4738 result."""
    held = [per_seed[s] for s in per_seed if int(s) in HELD and isinstance(per_seed[s], dict)]
    if not held:
        return {}

    def merge(vs):
        if all(isinstance(v, dict) for v in vs):
            return {k: merge([v[k] for v in vs if k in v]) for k in vs[0]}
        if all(isinstance(v, (int, float)) and v is not None for v in vs):
            return float(np.mean(vs)) if len(vs) else None
        return vs[0]
    return merge(held)


def lawsel(model, laws):
    return [l for l in laws if l in model.laws]


# ---------------------------------------------------------------- groups

def g_bias(model, dev, seed, quick):
    """Window cut from a large rollout vs isolated near-field truth: mean signed force difference (hole 4 bias check)."""
    if "grav" not in model.laws:
        return "skipped: law grav"
    n = 1500 if quick else 4096
    b = grav_batch("clumpy", n, 256.0, seed, 1, dev)
    r = truth_roll(b, model.dt, 10)
    P, V, M = r["pos"][0, 10], r["vel"][0, 10], b["mass"][0]
    rng = np.random.default_rng(seed)
    res, sgn, dens, farproj = [], [], [], []
    for _ in range(8 if quick else 24):
        c = P[int(rng.integers(0, n))]
        rel = P - c
        inw = rel.abs().amax(-1) < model.reach / 2 if model.reach < 1e9 else rel.abs().amax(-1) < 32
        if int(inw.sum()) < 8:
            continue
        x, v, m = rel[inw][None], V[inw][None], M[inw][None]
        k1 = torch.ones_like(m)
        a_iso = tt.grav_acc(x, m, k1)
        far = tt.far_expansion(torch.zeros(1, 2, device=dev, dtype=torch.float64), (P - c), M, (~inw).to(torch.float64)[None])
        a_far = tt.ExtField(*far).acc(x)
        st = model.init_state(1, dev)
        a_m, _ = model.force(x, v, m, k1, st, dict(phys=tt.Physics(grav=dict(eps=tt.EPS)), ext=None))
        d = (a_m - a_iso)[0]
        dens.append(float(inw.sum()))
        res.append(float(d.norm(dim=-1).pow(2).sum().sqrt() / a_iso[0].norm(dim=-1).pow(2).sum().sqrt().clamp_min(1e-12)))
        sgn.append(d.mean(0).cpu().numpy())
        u = a_far[0] / a_far[0].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        farproj.append(float((d * u).sum(-1).mean()))
    sgn = np.array(sgn)
    se = sgn.std(0, ddof=1) / math.sqrt(len(sgn)) if len(sgn) > 1 else np.full(2, np.nan)
    return dict(windows=len(res), rel_rms_force_err=float(np.mean(res)), mean_signed_diff=sgn.mean(0).tolist(), se=se.tolist(),
                z=(sgn.mean(0) / np.where(se > 0, se, np.nan)).tolist(), mean_diff_along_far_dir=float(np.mean(farproj)),
                corr_rel_err_vs_n=float(np.corrcoef(res, dens)[0, 1]) if len(res) > 2 else None,
                verdict="pass if |z|<2 on both axes (consistent with zero within window noise)")


def g_seams(model, dev, seed, quick):
    if "grav" not in model.laws:
        return "skipped: law grav"
    tl = ti.Tiled(model, 64.0, sparse=False) if model.reach < 1e9 else model
    rng = np.random.default_rng(seed)
    out = {}
    jm, jt, im, it = [], [], [], []
    xs = np.linspace(-6, 6, 49)
    mid = len(xs) // 2 - 1
    for rep in range(4 if quick else 12):
        nb = 200
        bp = np.concatenate([rng.uniform([-26, -30], [-8, -12], (nb // 2, 2)), rng.uniform([8, -30], [26, -12], (nb // 2, 2))])
        bm = np.exp(rng.uniform(np.log(0.05), np.log(1.0), nb))
        for y in (-3.0, 0.0, 3.0):
            probes = np.stack([xs, np.full_like(xs, y)], 1)
            ctl = probes + np.array([-16.0, 0.0])
            pos = torch.as_tensor(np.concatenate([bp, probes, ctl]), device=dev)[None]
            mass = torch.as_tensor(np.concatenate([bm, np.full(2 * len(xs), 1e-7)]), device=dev)[None]
            mk = torch.ones_like(mass)
            a_t = tt.grav_acc(pos, mass, mk)[0]
            a_m = tl.force(pos, torch.zeros_like(pos), mass, mk, tl.init_state(1, dev), dict(phys=tt.Physics(grav=dict(eps=tt.EPS))))[0][0]
            for sl, lst in ((slice(nb, nb + len(xs)), "seam"), (slice(nb + len(xs), nb + 2 * len(xs)), "ctl")):
                am, at_ = a_m[sl].cpu().numpy(), a_t[sl].cpu().numpy()
                if lst == "seam":
                    scale = float(np.linalg.norm(at_, axis=1).mean()) + 1e-12
                    lm, lt = np.linalg.norm(np.diff(am, axis=0), axis=1), np.linalg.norm(np.diff(at_, axis=0), axis=1)
                    rest = np.delete(np.arange(len(lm)), [mid - 1, mid, mid + 1])
                    jm.append(lm[mid] / scale), jt.append(lt[mid] / scale), im.append(np.median(lm[rest]) / scale), it.append(np.median(lt[rest]) / scale)
    out["seam_jump_over_field_model"], out["seam_jump_over_field_truth"] = float(np.mean(jm)), float(np.mean(jt))
    out["interior_link_over_field_model"], out["interior_link_over_field_truth"] = float(np.mean(im)), float(np.mean(it))
    out["seam_jump_ratio_model"] = float(np.mean(jm) / max(np.mean(im), 1e-3 * 1.0))
    out["seam_jump_ratio_truth"] = float(np.mean(jt) / max(np.mean(it), 1e-3))
    cl = grav_batch("plummer", 300, 40.0, seed, 4, dev)
    a_t = tt.grav_acc(cl["pos"], cl["mass"], cl["mask"])
    tl2 = ti.Tiled(model, 64.0, sparse=False) if model.reach < 1e9 else model
    a_tl = torch.stack([tl2.force(cl["pos"][b:b + 1], cl["vel"][b:b + 1], cl["mass"][b:b + 1], cl["mask"][b:b + 1],
                                  tl2.init_state(1, dev), dict(phys=cl["phys"]))[0][0] for b in range(4)])
    st = model.init_state(4, dev)
    a_sg = model.force(cl["pos"], cl["vel"], cl["mass"], cl["mask"], st, dict(phys=cl["phys"]))[0]
    rel = lambda a: float(((a - a_t) ** 2).sum().sqrt() / (a_t ** 2).sum().sqrt())
    out["tiled_vs_exact_rel_rms"], out["single_vs_exact_rel_rms"] = rel(a_tl), rel(a_sg)
    dist = (cl["pos"].abs().amin(-1)).cpu().numpy()
    e = (a_tl - a_t).norm(dim=-1).cpu().numpy() / (a_t.norm(dim=-1).cpu().numpy() + 1e-12)
    out["tiled_err_by_dist_to_edge"] = {lab: float(np.median(e[(dist >= lo) & (dist < hi)])) if ((dist >= lo) & (dist < hi)).any() else None
                                        for lab, lo, hi in (("<2", 0, 2), ("2-6", 2, 6), (">6", 6, 1e9))}
    return out


def g_range(model, dev, seed, quick):
    if "grav" not in model.laws:
        return "skipped: law grav"
    tl = wrap(model, 256.0)
    seps = [2, 4, 8, 16, 32, 64, 128] if not quick else [4, 32, 128]
    out = {"two_body": {}, "satellite_ring": {}, "arms": "A/B/C comparison needs the tiered models (not available to a single-tier baseline)"}
    for s in seps:
        pos = torch.tensor([[[-s / 2, 0.0], [s / 2, 0.0]]], device=dev, dtype=torch.float64)
        mass = torch.tensor([[5.0, 5.0]], device=dev, dtype=torch.float64)
        mk = torch.ones_like(mass)
        a_t = tt.grav_acc(pos, mass, mk)
        a_m = tl.force(pos, torch.zeros_like(pos), mass, mk, tl.init_state(1, dev), dict(phys=tt.Physics(grav=dict(eps=tt.EPS))))[0]
        out["two_body"][str(s)] = dict(rel_err=float((a_m - a_t).norm() / a_t.norm()), mag_ratio=float(a_m[0].norm() / a_t[0].norm()))
    rng = np.random.default_rng(seed)
    p, v, m = sc.big_scene("plummer", 300, 24.0, rng, dev)
    rs = [4, 8, 16, 32, 64, 100] if not quick else [8, 64]
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for r in rs:
        sat = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        pos = torch.as_tensor(np.concatenate([np.broadcast_to(p, (8,) + p.shape), sat[:, None]], 1), device=dev)
        mass = torch.as_tensor(np.concatenate([np.broadcast_to(m, (8,) + m.shape), np.ones((8, 1))], 1), device=dev)
        mk = torch.ones_like(mass)
        a_t = tt.grav_acc(pos, mass, mk)[:, -1]
        a_m = tl.force(pos, torch.zeros_like(pos), mass, mk, tl.init_state(8, dev), dict(phys=tt.Physics(grav=dict(eps=tt.EPS))))[0][:, -1]
        out["satellite_ring"][str(r)] = dict(rel_err=float((a_m - a_t).norm() / a_t.norm()), mag_ratio=float(a_m.norm(dim=-1).mean() / a_t.norm(dim=-1).mean()))
    return out


def g_gravity(model, dev, seed, quick):
    if "grav" not in model.laws:
        return "skipped: law grav"
    steps = 20 if quick else 100
    out = {}
    for ic, n in (("plummer", 200), ("cold_collapse", 200), ("binary", 60), ("disc", 300), ("bh", 300), ("multiclump", 300)):
        span = 128.0 if ic == "multiclump" else 44.0
        b = grav_batch(ic, n, span, seed, 4 if not quick else 2, dev)
        tl = wrap(model, 128.0 if ic == "multiclump" else 64.0)
        out[ic] = compare(tl, b, steps, name=f"gravity_{ic}", extra_ctx=None)
    return out


def g_conservation(model, dev, seed, quick):
    out = {}
    if "grav" in model.laws:
        b = grav_batch("plummer", 200, 44.0, seed, 2, dev)
        out["gravity_long"] = compare(wrap(model, 64.0), b, 60 if quick else 500, ks=(10, 50, 100, 200, 500))
    if "lj" in model.laws:
        for ph in ("lj_solid", "lj_liquid"):
            rng = np.random.default_rng(seed)
            rho, T, jit = sc.LJ_PHASE[ph]
            a = sc.lj_spacing(rho)
            nx = 18
            pts = sc.hex_lattice(nx, int(nx * 2 / math.sqrt(3)), a)
            box = np.array([nx * a, int(nx * 2 / math.sqrt(3)) * a * math.sqrt(3) / 2])
            pts = pts + rng.normal(0, jit * a, pts.shape)
            n = len(pts)
            s = sc._empty(ph, 0.1, 20)
            s.update(pos=pts, vel=sc.maxwell(n, T, rng), mass=np.ones(n), kmat=np.ones(n), law=dict(lj=dict(rc=2.5)),
                     bound=dict(kind="periodic", period=box), tags=dict(equil=(T, 200)))
            b = sc.equilibrate(sc.build_batch([s], dev), 200)
            out[ph] = compare(model, b, 20 if quick else 160, ks=(10, 20, 50, 100, 160))
    if not out:
        return "skipped: no supported law"
    return out


def timed_step(model, batch, reps=3):
    st = model.init_state(batch["pos"].shape[0], batch["pos"].device)
    ctx = dict(ext=None, t=0.0, phys=batch["phys"])
    pos, vel = batch["pos"], batch["vel"]
    for _ in range(2):
        pos, vel, st = ti.advance(model, pos, vel, batch["mass"], batch["mask"], st, ctx)
    cuda = pos.is_cuda
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        pos, vel, st = ti.advance(model, pos, vel, batch["mass"], batch["mask"], st, ctx)
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1e3, (torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0)


def fit_exp(ns, ts):
    if len(ns) < 3:
        return None
    return float(np.polyfit(np.log(ns), np.log(ts), 1)[0])


def g_sparsity(model, dev, seed, quick):
    if "grav" not in model.laws:
        return "skipped: law grav"
    span = 256.0
    rng = np.random.default_rng(seed)
    out = {"n_sweep": {}, "empty_sweep": {}}
    ns = [250, 500, 1000, 2000, 4000] if not quick else [250, 1000]
    for mode in ("dense", "sparse"):
        tl = wrap(model, span, sparse=(mode == "sparse"))
        ts, ms = [], []
        out["n_sweep"][mode] = {}
        for n in ns:
            pos = rng.uniform(-span / 2, span / 2, (1, n, 2)) * 0.95
            b = dict(pos=torch.as_tensor(pos, device=dev), vel=torch.as_tensor(rng.normal(0, 0.3, (1, n, 2)), device=dev),
                     mass=torch.full((1, n), 0.05, device=dev, dtype=torch.float64), mask=torch.ones(1, n, device=dev, dtype=torch.float64),
                     phys=tt.Physics(grav=dict(eps=tt.EPS)))
            t, mem = timed_step(tl, b)
            ts.append(t), ms.append(mem)
            out["n_sweep"][mode][str(n)] = dict(ms_per_step=t, peak_mb=mem)
        out["n_sweep"][mode]["time_exponent"] = fit_exp(ns, ts)
        out["n_sweep"][mode]["mem_exponent"] = fit_exp(ns, ms)
        out["empty_sweep"][mode] = {}
        for f in ([1.0, 0.5, 0.25, 0.125, 0.0625] if not quick else [1.0, 0.125]):
            w = span * math.sqrt(f) / 2 * 0.95
            n = 2000 if not quick else 500
            pos = rng.uniform(-w, w, (1, n, 2))
            b = dict(pos=torch.as_tensor(pos, device=dev), vel=torch.as_tensor(rng.normal(0, 0.3, (1, n, 2)), device=dev),
                     mass=torch.full((1, n), 0.05, device=dev, dtype=torch.float64), mask=torch.ones(1, n, device=dev, dtype=torch.float64),
                     phys=tt.Physics(grav=dict(eps=tt.EPS)))
            t, mem = timed_step(tl, b)
            act = getattr(tl, "last_active", None)
            out["empty_sweep"][mode][str(f)] = dict(ms_per_step=t, peak_mb=mem, active_tiles=act, tiles=getattr(tl, "last_tiles", None))
    b = grav_batch("cold_collapse", 300, 44.0, seed, 2, dev)
    r = truth_roll(b, model.dt, 60 if not quick else 15)
    dens = []
    for k in (0, 15, 30, 60):
        if k < r["pos"].shape[1]:
            d = torch.cdist(r["pos"][0, k], r["pos"][0, k])
            dens.append((k, float((d < 1.0).sum(-1).float().mean())))
    out["collapsing_core"] = dict(truth_mean_neighbours_within_1=dens, **compare(wrap(model, 64.0), b, 60 if not quick else 15, name="sparsity_cold_collapse"))
    return out


def frag_count(pos, mask, thr=1.4):
    p = pos[mask > 0]
    d = torch.cdist(p, p) < thr
    lab = torch.arange(len(p), device=p.device)
    for _ in range(len(p)):
        new = torch.where(d, lab[None, :], lab.new_full((), len(p))).amin(1)
        new = torch.minimum(new, lab)
        if torch.equal(new, lab):
            break
        lab = new
    return int(lab.unique().numel())


def ke_rel(r, mass, mask, split):
    """Kinetic energy in the centre-of-mass frame of the two bodies' relative motion: sum over bodies of 0.5 M_body |V_body - V_cm|^2."""
    v = r["vel"] * 1.0
    m = (mass * mask)[:, None, :, None]
    out = []
    for k in (0, -1):
        vk = v[:, k]
        a = (m[:, 0, :split] * vk[:, :split]).sum(1) / m[:, 0, :split].sum(1)
        b_ = (m[:, 0, split:] * vk[:, split:]).sum(1) / m[:, 0, split:].sum(1)
        ma, mb = m[:, 0, :split].sum((1, 2)), m[:, 0, split:].sum((1, 2))
        vcm = (ma[:, None] * a + mb[:, None] * b_) / (ma + mb)[:, None]
        out.append(0.5 * (ma * ((a - vcm) ** 2).sum(-1) + mb * ((b_ - vcm) ** 2).sum(-1)))
    return out


def obj_batch(dev, seed, fn, B=4, **kw):
    rng = np.random.default_rng(seed)
    scenes = [getattr(sc, fn)(rng, 256, **kw) for _ in range(B)]
    return scenes, sc.build_batch(scenes, dev)


def g_objects(model, dev, seed, quick):
    if not ({"contact", "bonded"} & set(model.laws)):
        return "skipped: needs contact/bonded law"
    out = {}
    steps = 20 if quick else 100
    if "bonded" in model.laws:
        _, b = obj_batch(dev, seed, "bonded_scene")
        b["vel"] = torch.zeros_like(b["vel"])
        out["rest_stability"] = compare(model, b, steps, name="objects_rest_stability")
        _, b = obj_batch(dev, seed, "spin_scene")
        out["spin"] = compare(model, b, steps, name="objects_spin")
        _, b = obj_batch(dev, seed, "pile_scene", B=3)
        out["pile"] = compare(model, b, steps, name="objects_pile")
        _, b = obj_batch(dev, seed, "fracture_scene", B=3)
        res = compare(model, b, steps, name="objects_fracture")
        rt, rm = truth_roll(b, model.dt, steps), model_roll(model, b, steps)
        res["fragments"] = dict(truth=[frag_count(rt["pos"][i, -1], b["mask"][i]) for i in range(3)], model=[frag_count(rm["pos"][i, -1], b["mask"][i]) for i in range(3)],
                                before=[frag_count(rt["pos"][i, 0], b["mask"][i]) for i in range(3)])
        out["fracture"] = res
        out["collision_sweep"] = {}
        for typ in ("bonded", "lj_droplet"):
            if typ == "lj_droplet" and "lj" not in model.laws:
                continue
            for bf in (0.0, 0.5):
                for v in ((1.0, 3.0) if quick else (1.0, 2.0, 4.0)):
                    scenes, b = obj_batch(dev, seed, "collision_scene", B=1, typ=typ, b_frac=bf, v_rel=v, om=[0.0, 0.0])
                    rt, rm = truth_roll(b, model.dt, steps), model_roll(model, b, steps)
                    ec = err_curve(rm["pos"], rt["pos"], b["mask"])
                    res = dict(err=at(ec), fragments=dict(truth=[frag_count(rt["pos"][i, -1], b["mask"][i]) for i in range(len(scenes))],
                                                          model=[frag_count(rm["pos"][i, -1], b["mask"][i]) for i in range(len(scenes))]))
                    res["momentum_err"] = float(((rm["vel"][:, -1] * (b["mass"] * b["mask"])[..., None]).sum(1) - (rt["vel"][:, -1] * (b["mass"] * b["mask"])[..., None]).sum(1)).norm(dim=-1).mean())
                    if len(scenes) and len({sc_["tags"]["split"] for sc_ in scenes}) == 1:
                        k0t, k1t = ke_rel(rt, b["mass"], b["mask"], scenes[0]["tags"]["split"])
                        k0m, k1m = ke_rel(rm, b["mass"], b["mask"], scenes[0]["tags"]["split"])
                        res["restitution_ke_ratio"] = dict(truth=float((k1t / k0t).mean()), model=float((k1m / k0m.clamp_min(1e-12)).mean()))
                    out["collision_sweep"][f"{typ}_b{bf}_v{v}"] = res
    if "contact" in model.laws:
        b = tier0_batch("contact", seed, 8, dev, n_max=40)
        out["contact_balls"] = compare(model, b, steps, name="objects_contact")
        out["tunnelling"] = {}
        for v in (2.0, 6.0, 12.0):
            s = sc._empty("tunnel", 0.15, 8)
            s.update(pos=np.array([[-6.0, 0.0], [6.0, 0.05]]), vel=np.array([[v, 0.0], [-v, 0.0]]), mass=np.ones(2), kmat=np.ones(2),
                     law=dict(contact=dict(radius=sc.BALL_R, k=sc.BALL_K)))
            bb = sc.build_batch([s], dev)
            rt, rm = truth_roll(bb, model.dt, 20), model_roll(model, bb, 20)
            gap = lambda r: float((r["pos"][0, :, 1, 0] - r["pos"][0, :, 0, 0]).min())
            out["tunnelling"][str(v)] = dict(min_gap_truth=gap(rt), min_gap_model=gap(rm), passed_through_model=bool(gap(rm) < 0))
    return out or "skipped: no supported law"


def g_boundaries(model, dev, seed, quick):
    out = {}
    steps = 20 if quick else 60
    for law, base in (("grav", "grav"), ("lj", "lj_gas"), ("contact", "contact")):
        if law not in model.laws:
            continue
        for bk in sc.BOUND_KINDS:
            rng = np.random.default_rng(seed)
            scenes = [sc.boundary_scene(rng, 64, bkind=bk, base=base) for _ in range(6)]
            b = sc.build_batch(scenes, dev)
            res = compare(model, b, steps, name=f"boundaries_{base}_{bk}")
            rt, rm = truth_roll(b, model.dt, steps), model_roll(model, b, steps, None)
            if bk != "periodic":
                lo = torch.as_tensor(np.stack([s["bound"]["lo"] for s in scenes]), device=dev)[:, None]
                hi = torch.as_tensor(np.stack([s["bound"]["hi"] for s in scenes]), device=dev)[:, None]
                esc = lambda r: float((((r["pos"][:, -1] < lo - 1.0) | (r["pos"][:, -1] > hi + 1.0)).any(-1).float() * b["mask"]).sum() / b["mask"].sum())
                res["escaped_fraction"] = dict(truth=esc(rt), model=esc(rm))
            out[f"{base}_{bk}"] = res
    return out or "skipped: no supported law"


def g_calibration(model, dev, seed, quick):
    if not hasattr(model, "error"):
        return "skipped: model has no error estimate (optional interface method error())"
    b = grav_batch("plummer", 200, 44.0, seed, 4, dev)
    st = model.init_state(4, dev)
    pred = model.error(b["pos"], b["vel"], b["mass"], b["mask"], st, dict(phys=b["phys"]))
    a_m = model.force(b["pos"], b["vel"], b["mass"], b["mask"], st, dict(phys=b["phys"]))[0]
    act = (a_m - tt.grav_acc(b["pos"], b["mass"], b["mask"])).norm(dim=-1)
    x, y = pred.flatten().cpu().numpy(), act.flatten().cpu().numpy()
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    return dict(spearman=float(np.corrcoef(rk(x), rk(y))[0, 1]), slope=float(np.polyfit(x, y, 1)[0]), mean_pred=float(x.mean()), mean_actual=float(y.mean()))


def g_exposure(model, dev, seed, quick):
    """Teacher-forced vs own-rollout force error (hole 5 diagnostic). For tier>=1 models with a .lower hook the residual error is used."""
    if "grav" not in model.laws:
        return "skipped: law grav"
    steps = 20 if quick else 50
    b = grav_batch("plummer", 200, 44.0, seed, 4, dev)
    tl = wrap(model, 64.0)
    rt, rm = truth_roll(b, model.dt, steps), model_roll(tl, b, steps)
    res = {}
    for nm, traj in (("truth_states", rt), ("model_states", rm)):
        errs = []
        for k in range(0, steps + 1, max(1, steps // 5)):
            p, v = traj["pos"][:, k], traj["vel"][:, k]
            a_t = tt.grav_acc(p, b["mass"], b["mask"])
            a_m = tl.force(p, v, b["mass"], b["mask"], tl.init_state(p.shape[0], dev), dict(phys=b["phys"]))[0]
            errs.append(float(((a_m - a_t) ** 2).sum((1, 2)).sqrt().div((a_t ** 2).sum((1, 2)).sqrt()).mean()))
        res[nm] = dict(zip([str(k) for k in range(0, steps + 1, max(1, steps // 5))], errs))
    res["gap_ratio_last"] = list(res["model_states"].values())[-1] / max(list(res["truth_states"].values())[-1], 1e-12)
    res["note"] = "memoryless force (recurrent field reset each call); tier>=1 residual form needs a model exposing .lower"
    return res


def g_generalisation(model, dev, seed, quick):
    if "grav" not in model.laws:
        return "skipped: law grav"
    out = {}
    steps = 10 if quick else 20
    for nm, (ic, n, span) in dict(unseen_N_800=("plummer", 800, 44.0), unseen_N_1500=("clumpy", 1500, 44.0), domain_x1p5=("plummer", 200, 66.0),
                                  few_bodies=("binary", 6, 30.0)).items():
        b = grav_batch(ic, n, span, seed, 2, dev)
        out[nm] = compare(wrap(model, 64.0 if span <= 50 else 2 * span), b, steps)
    b = grav_batch("plummer", 200, 40.0, seed, 2, dev)
    big = grav_batch("plummer", 200, 40.0, seed + 1, 2, dev)
    big["mass"] = big["mass"] * torch.exp(torch.linspace(-4, 4, 200, device=dev, dtype=torch.float64))[None]
    out["mass_ratio_3e3"] = compare(wrap(model, 64.0), big, steps)
    F = lambda p: model.force(p, b["vel"], b["mass"], b["mask"], model.init_state(2, dev), dict(phys=b["phys"]))[0]
    f0 = F(b["pos"])
    cell = getattr(model, "cell", 0.5)
    sh = {}
    for frac in (0.1, 0.25, 0.5):
        d = torch.tensor([frac * cell, 0.61 * frac * cell], device=dev, dtype=torch.float64)
        sh[str(frac)] = float(((F(b["pos"] + d) - f0) ** 2).sum().sqrt() / (f0 ** 2).sum().sqrt())
    out["shift_invariance_force_change_per_cell_fraction"] = sh
    R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=dev, dtype=torch.float64)
    out["rot90_equivariance_rel_err"] = float(((F(b["pos"] @ R.T) - f0 @ R.T) ** 2).sum().sqrt() / (f0 ** 2).sum().sqrt())
    return out


def g_timescale(model, dev, seed, quick):
    if "multirate" not in model.caps:
        return "skipped: model has no multirate capability (ctx['multirate'] = m slow-tier substep ratio)"
    out = {}
    for m in (1, 2, 4):
        b = grav_batch("plummer", 200, 44.0, seed, 2, dev)
        out[f"m={m}"] = compare(wrap(model, 64.0), b, 20 if quick else 100, extra_ctx=dict(multirate=m))
    return out


RUN = dict(bias=g_bias, seams=g_seams, range=g_range, gravity=g_gravity, conservation=g_conservation, sparsity=g_sparsity,
           objects=g_objects, boundaries=g_boundaries, calibration=g_calibration, exposure=g_exposure, generalisation=g_generalisation,
           timescale=g_timescale)


# ---------------------------------------------------------------- rendering

def density(pos, mask, extent, G=96):
    p = pos[mask > 0]
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=G, range=[[-extent / 2, extent / 2]] * 2)
    return h.T


GRID_STEPS = (0, 1, 3, 8, 20, 35, 50, 75, 100)


def render_grid(name, frames, path, steps=GRID_STEPS, extent=None):
    """Rows truth / model / model-truth density, per-frame autoscale; columns run to step 100 (the last available step if shorter).
    Spatial extent follows the truth's 98th-percentile radius over all frames unless given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Pt, Pm, m, mk = frames
    last = len(Pt) - 1
    steps = sorted({min(s, last) for s in steps})
    live = mk > 0
    if extent is None:
        r = np.percentile(np.abs(np.concatenate([Pt[:, live].reshape(-1, 2), Pm[:, live].reshape(-1, 2)])), 98)
        extent = float(min(max(2.2 * r, 8.0), 400.0))
    fig, ax = plt.subplots(3, len(steps), figsize=(1.9 * len(steps), 6.2), squeeze=False)
    rows = ["truth", "model", "model-truth"]
    for j, s in enumerate(steps):
        dt_, dm_ = density(Pt[s], mk, extent), density(Pm[s], mk, extent)
        for i, img in enumerate((dt_, dm_, dm_ - dt_)):
            ax[i, j].imshow(img, origin="lower", cmap="magma" if i < 2 else "coolwarm", vmin=None if i < 2 else -abs(img).max() - 1e-9,
                            vmax=None if i < 2 else abs(img).max() + 1e-9)
            ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(f"step {s}", fontsize=8)
            if j == 0:
                ax[i, j].set_ylabel(rows[i], fontsize=8)
    fig.suptitle(f"{name} (view +-{extent / 2:.0f} units)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_model(spec, dev, dt):
    if spec == "oracle":
        return ti.Oracle(dt)
    if spec == "constvel":
        return ti.ConstVel(dt)
    if spec.startswith("scatter:"):
        return ti.load_scatter(spec[8:], dev, dt, name="scatterE")
    from scripts import tiered_adapters as ta

    kind, _, rest = spec.partition(":")
    loaders = dict(tiered=lambda: ta.load_tiered(rest, dev), t0ext=lambda: ta.load_t0ext(rest, dev, dt), wide=lambda: ta.load_wide(rest, dev, dt),
                   tiled=lambda: ta.load_tiled(rest, dev, dt))
    if kind in loaders:
        return loaders[kind]()
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="oracle | constvel | scatter:<ckpt> | tiered:<nets>[,<tier0>] | t0ext:<ckpt> | wide:<ckpt>[,grid=,extent=] | tiled:<ckpt>[,nt=,tile=,halo=,far=]")
    ap.add_argument("--groups", default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[4738, *HELD])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out", default="results/tiered/eval.json")
    ap.add_argument("--grid-dir", default="results/tiered/grids")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = load_model(args.model, dev, args.dt)
    groups = GROUPS if args.groups == "all" else args.groups.split(",")
    res = dict(model=model.name, spec=args.model, laws=sorted(model.laws), caps=sorted(model.caps), reach=model.reach if model.reach < 1e9 else "inf",
               seeds=args.seeds, quick=args.quick, groups={})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.grid_dir, exist_ok=True)
    for g in groups:
        t0 = time.time()
        per = {}
        status = "ok"
        for s in args.seeds:
            r = RUN[g](model, dev, s, args.quick)
            if isinstance(r, str):
                status = r
                break
            per[str(s)] = r
            if s == args.seeds[0]:
                for nm, fr in list(FRAMES.items()):
                    render_grid(f"{model.name} {nm} seed {s}", fr, os.path.join(args.grid_dir, f"{model.name}_{nm}.png"))
            FRAMES.clear()
        res["groups"][g] = dict(status=status, seconds=round(time.time() - t0, 1), seeds=per, summary=agg(per) if status == "ok" else {})
        print(f"[{g}] {status} {time.time() - t0:.1f}s", flush=True)
        json.dump(tolist(res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
