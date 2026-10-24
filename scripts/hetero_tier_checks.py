"""Hole-4 / hole-5 / hole-7 baseline measurements for the heterogeneous tiered design (docs/theory/heterogeneous-tiered.md).
Uses the existing Exp E checkpoint (trained on isolated scenes of extent 64, i.e. the hole-3 'scenes no larger than reach' regime) and TiledScatter.
Phases: a (window-cutout bias), b (teacher-forced vs rollout residual gap), c (seam crossing: error jump and energy gain).
Usage: PYTHONPATH=. python scripts/hetero_tier_checks.py --ckpt E.pt --kernel file:kern/r1b_mlp.pt --phase a --out results/hetero_a.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_scaling_ops as ops
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling import TiledScatter
from scripts.scatter_tiling_eval import DT, EPS, SUB, class_masks, load, make_tiled, single_setup, VARIANTS

SEEDS = (4738, 9000, 12000)
REPS = 3
HALF = 32.0
BINS = ((0, 4), (4, 8), (8, 16), (16, 32))
SCENES_A = {"sparse": (800, 60.0, 256.0, ((0.0, 0.0), (40.0, 0.0), (0.0, 40.0), (-40.0, -40.0))),
            "dense": (1200, 16.0, 128.0, ((0.0, 0.0), (20.0, 0.0), (0.0, 20.0), (-20.0, -20.0)))}


def seeds_of(a):
    return SEEDS[:1] if a.smoke else SEEDS


def reps_of(a):
    return 1 if a.smoke else REPS


def mforce(model, pos, vel, field):
    n = pos.shape[0]
    one = torch.ones(1, n, device=pos.device)
    a, f = model.force(pos[None], vel[None], one, one, field)
    return a[0], f


@torch.no_grad()
def phase_a(model, dev, a):
    ticks = (0, 5, 10, 20)
    rows = []
    for sname, (nb, sig, ext, centres) in SCENES_A.items():
      for seed in seeds_of(a):
        for rep in range(reps_of(a)):
            p0, v0 = make_scene(nb, sig, 1.0, ext, seed, rep)
            P0, V0 = torch.tensor(p0, device=dev), torch.tensor(v0, device=dev)
            Pt, Vt = ops.exact_rollout(P0, V0, 20, DT, SUB, EPS)
            for ci, c in enumerate(centres):
                cc = torch.tensor(c, device=dev, dtype=torch.float64)
                ops.retarget(model, 128, 2 * HALF)
                field = model.init_field(1, dev)[0]
                for t in range(21):
                    rel = Pt[t] - cc
                    idx = (rel.abs() < HALF).all(-1).nonzero()[:, 0]
                    pw, vw = rel[idx], Vt[t][idx]
                    am, field = mforce(model, pw.float(), vw.float(), field)
                    if t not in ticks:
                        continue
                    am = am.double()
                    t_iso = ops.exact_acc(pw, EPS)
                    t_full = ops.exact_acc(Pt[t], EPS)[idx]
                    F = t_full - t_iso
                    F = F - F.mean(0, keepdim=True)
                    t_fullm = t_iso + F
                    d_iso = am - t_iso
                    d_full = am - t_fullm
                    edge = HALF - pw.abs().max(-1).values
                    for bi, (lo, hi) in enumerate(BINS):
                        m = (edge >= lo) & (edge < hi)
                        if not m.any():
                            continue
                        rows.append({
                            "scene": sname, "seed": seed, "rep": rep, "centre": ci, "tick": t, "bin": bi, "n": int(m.sum()),
                            "dx": float(d_iso[m, 0].sum()), "dy": float(d_iso[m, 1].sum()),
                            "Fx": float(F[m, 0].sum()), "Fy": float(F[m, 1].sum()),
                            "num": float((d_iso[m] * F[m]).sum()), "den": float((F[m] ** 2).sum()),
                            "e_iso": float(d_iso[m].norm(dim=-1).sum()), "a_iso": float(t_iso[m].norm(dim=-1).sum()),
                            "e_full": float(d_full[m].norm(dim=-1).sum()), "a_full": float(t_fullm[m].norm(dim=-1).sum()),
                            "F": float(F[m].norm(dim=-1).sum())})
                rel0 = Pt[0] - cc
                idx0 = (rel0.abs() < HALF).all(-1).nonzero()[:, 0]
                ps, vs = rel0[idx0], Vt[0][idx0]
                n = len(idx0)
                Pi, Vi = ops.exact_rollout(ps, vs, 20, DT, SUB, EPS)
                ops.retarget(model, 128, 2 * HALF)
                one = torch.ones(1, n, device=dev)
                st = (model.init_field(1, dev)[0], None)
                p, v = ps.float()[None], vs.float()[None]
                Pm, Vm = [p[0]], [v[0]]
                for _ in range(20):
                    p, v, st, _ = model.step(p, v, one, one, st)
                    Pm.append(p[0])
                    Vm.append(v[0])
                Pm, Vm = torch.stack(Pm).double(), torch.stack(Vm).double()
                edge0 = HALF - ps.abs().max(-1).values
                e0 = ops.exact_energy(Pi[0], Vi[0], EPS)
                for k in (5, 10, 20):
                    ei = Pm[k] - Pi[k]
                    ef = Pm[k] - (Pt[k][idx0] - cc)
                    for bi, (lo, hi) in enumerate(BINS):
                        m = (edge0 >= lo) & (edge0 < hi)
                        if not m.any():
                            continue
                        rows.append({
                            "scene": sname, "seed": seed, "rep": rep, "centre": ci, "tick": k, "bin": bi, "kind": "pos", "n": int(m.sum()),
                            "pi": float(ei[m].norm(dim=-1).sum()), "pf": float(ef[m].norm(dim=-1).sum()),
                            "pix": float(ei[m, 0].sum()), "piy": float(ei[m, 1].sum()),
                            "pfx": float(ef[m, 0].sum()), "pfy": float(ef[m, 1].sum())})
                    rows.append({
                        "scene": sname, "seed": seed, "rep": rep, "centre": ci, "tick": k, "kind": "energy", "n": n,
                        "E_model": ops.exact_energy(Pm[k], Vm[k], EPS) - e0,
                        "E_iso": ops.exact_energy(Pi[k], Vi[k], EPS) - e0, "ke0": float(0.5 * (Vi[0] ** 2).sum())})
            print("a done", sname, seed, rep, flush=True)
    return rows


def spec_of(name):
    return VARIANTS[name]


def variant_runner(model, name, dev):
    """returns (force_tf(pos, vel, field)->(a, field), init_field, step_fn(p, v, st)->(p, v, st, a), classes_fn)"""
    n_box = None
    if name == "single":
        T = make_tiled(model, VARIANTS["g16_fx"])

        def init():
            return model.init_field(1, dev)[0]

        def tf(p, v, f):
            return mforce(model, p, v, f)

        def step(p, v, mass, st):
            one = torch.ones(1, p.shape[0], device=dev)
            pn, vn, st2, _ = model.step(p[None], v[None], mass[None], one, st)
            return pn[0], vn[0], st2, st2[1][0]

        def st0():
            return (model.init_field(1, dev)[0], None)
    else:
        T = make_tiled(model, VARIANTS[name])

        def init():
            return T.init_field(dev)

        def tf(p, v, f):
            return T.force(p, v, torch.ones(p.shape[0], device=dev), f)

        def step(p, v, mass, st):
            pn, vn, st2 = T.step(p, v, mass, st)
            return pn, vn, st2, st2[1]

        def st0():
            return (T.init_field(dev), None)
    return T, init, tf, step, st0


def stat_res(a_m, a_x, P, T, cen):
    R = a_x - a_m
    ds, dout = T.seam_dist(P)
    mk = class_masks(ds, dout)
    out = {}
    for k in ("seam<8", "interior", "outer<8"):
        m = mk[k]
        out[k] = {"n": int(m.sum()), "rel": float(R[m].norm(dim=-1).sum() / a_x[m].norm(dim=-1).sum().clamp_min(1e-12)) if m.any() else None}
    allm = torch.ones_like(ds, dtype=torch.bool)
    rn = R.norm(dim=-1).sum()
    an = a_x.norm(dim=-1).sum().clamp_min(1e-12)
    out["all"] = {"n": int(allm.sum()), "rel": float(rn / an)}
    rv = P - cen
    rhat = rv / rv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out["inward_frac"] = float((-(R * rhat).sum(-1)).sum() / an)
    out["tang_frac"] = float((R[:, 0] * -rhat[:, 1] + R[:, 1] * rhat[:, 0]).sum() / an)
    cell = 8.0
    nc = int(128 / cell)
    ix = ((P[:, 0] + 64) / cell).floor().clamp(0, nc - 1).long()
    iy = ((P[:, 1] + 64) / cell).floor().clamp(0, nc - 1).long()
    key = ix * nc + iy
    cnt = torch.zeros(nc * nc, device=P.device, dtype=R.dtype).index_add_(0, key, torch.ones_like(R[:, 0]))
    sm = torch.zeros(nc * nc, 2, device=P.device, dtype=R.dtype).index_add_(0, key, R)
    smooth = (sm.norm(dim=-1)).sum()
    out["smooth_rel"] = float(smooth / an)
    out["smooth_over_R"] = float(smooth / rn.clamp_min(1e-12))
    return out


@torch.no_grad()
def phase_b(model, dev, a):
    names = ["single", "g16_fx", "g16_fx_far", "t32_g16_fx", "t32_g16_fx_far"]
    fams = {"cold300": (300, 8.0, 0.3), "warm1200": (1200, 16.0, 1.0)}
    steps = 40
    ticks = (5, 10, 20, 40)
    rows = []
    for fam, (n, sigma, vfac) in fams.items():
        for seed in seeds_of(a):
            for rep in range(reps_of(a)):
                p0, v0 = make_scene(n, sigma, vfac, 128.0, seed, rep)
                P0, V0 = torch.tensor(p0, device=dev), torch.tensor(v0, device=dev)
                Pt, Vt = ops.exact_rollout(P0, V0, steps, DT, SUB, EPS)
                mass = torch.ones(n, device=dev)
                cen = torch.zeros(2, device=dev)
                for name in names:
                    single_setup(model)
                    T, init, tf, step, st0 = variant_runner(model, name, dev)
                    f = init()
                    tfa = {}
                    for t in range(steps + 1):
                        am, f = tf(Pt[t].float(), Vt[t].float(), f)
                        if t in ticks:
                            tfa[t] = am
                    single_setup(model)
                    p, v = P0.float(), V0.float()
                    st = st0()
                    ro = {}
                    for t in range(1, steps + 1):
                        p, v, st, am = step(p, v, mass, st)
                        if t in ticks:
                            ro[t] = (p.clone(), am.clone())
                    single_setup(model)
                    for t in ticks:
                        ax = ops.exact_acc(Pt[t], EPS).float()
                        rec = {"fam": fam, "seed": seed, "rep": rep, "variant": name, "tick": t,
                               "tf": stat_res(tfa[t], ax, Pt[t].float(), T, cen)}
                        pr, ar = ro[t]
                        axr = ops.exact_acc(pr.double(), EPS).float()
                        rec["ro"] = stat_res(ar, axr, pr, T, cen)
                        rec["pos_err"] = float((pr.double() - Pt[t]).norm(dim=-1).mean())
                        rows.append(rec)
                print("b done", fam, seed, rep, flush=True)
    return rows


def phi_exact(pos, eps, elems=3.0e8):
    N = pos.shape[0]
    c = max(1, int(elems / N))
    out = torch.empty(N, dtype=pos.dtype, device=pos.device)
    for s in range(0, N, c):
        e = min(N, s + c)
        d = pos[None, :, :] - pos[s:e, None, :]
        inv = ((d ** 2).sum(-1) + eps ** 2) ** -0.5
        idx = torch.arange(s, e, device=pos.device)
        inv[idx - s, idx] = 0.0
        out[s:e] = -inv.sum(1)
    return out


def tile_id(T, pos):
    n = T.nt
    t = ((pos - T.X0) / T.Lt).floor().clamp(0, n - 1).long()
    return t[:, 0] * n + t[:, 1]


def sc(v):
    return None if v is None else float(v)


@torch.no_grad()
def phase_c(model, dev, a):
    steps = 40
    cfgs = {"t64": "g16_fx_far", "t32": "t32_g16_fx_far"}
    scenes = []
    for seed in seeds_of(a):
        for rep in range(reps_of(a)):
            scenes.append(("warm1200", seed, rep, make_scene(1200, 16.0, 1.0, 128.0, seed, rep)))
            p, v = make_scene(600, 6.0, 1.0, 128.0, seed, rep)
            p = p + np.array([-10.0, 0.0])
            v = v + np.array([3.0, 0.0])
            scenes.append(("drift600", seed, rep, (p, v)))
    rows = []
    for fam, seed, rep, (p0, v0) in scenes:
        n = p0.shape[0]
        P0, V0 = torch.tensor(p0, device=dev), torch.tensor(v0, device=dev)
        Pt, Vt = ops.exact_rollout(P0, V0, steps, DT, SUB, EPS)
        mass = torch.ones(n, device=dev)
        for cname, vname in cfgs.items():
            Tref = make_tiled(model, VARIANTS[vname])
            runs = {"exact": (Pt.float(), Vt.float())}
            tfres = {}
            for name in ("single", vname):
                single_setup(model)
                T, init, tf, step, st0 = variant_runner(model, name, dev)
                f = init()
                R = []
                for t in range(steps + 1):
                    am, f = tf(Pt[t].float(), Vt[t].float(), f)
                    R.append(am - ops.exact_acc(Pt[t], EPS).float())
                tfres[name] = R
                single_setup(model)
                p, v = P0.float(), V0.float()
                st = st0()
                Ps, Vs = [p], [v]
                for t in range(steps):
                    p, v, st, _ = step(p, v, mass, st)
                    Ps.append(p)
                    Vs.append(v)
                runs[name] = (torch.stack(Ps), torch.stack(Vs))
                single_setup(model)
            for rn, (P, V) in runs.items():
                ex, ctl, jx, jc, ae_x, ae_c = [], [], [], [], [], []
                dE = {}
                cr_n = ct_n = 0
                dcr, dct, acr, act = [], [], [], []
                E = [ops.exact_energy(P[t].double(), V[t].double(), EPS) for t in range(steps + 1)]
                e_prev = 0.5 * (V[0].double() ** 2).sum(-1) + phi_exact(P[0].double(), EPS)
                for t in range(steps):
                    e_next = 0.5 * (V[t + 1].double() ** 2).sum(-1) + phi_exact(P[t + 1].double(), EPS)
                    de = e_next - e_prev
                    e_prev = e_next
                    k0, k1 = tile_id(Tref, P[t]), tile_id(Tref, P[t + 1])
                    ds0, _ = Tref.seam_dist(P[t])
                    ds1, _ = Tref.seam_dist(P[t + 1])
                    cross = k0 != k1
                    ctrl = (~cross) & (ds0 < 4) & (ds1 < 4)
                    dcr.append(de[cross])
                    dct.append(de[ctrl])
                    if rn != "exact":
                        Rt = tfres[rn]
                        jump = (Rt[t + 1] - Rt[t]).norm(dim=-1)
                        # error jump measured on shared exact states: crossing flag from exact states
                        ke0, ke1 = tile_id(Tref, Pt[t]), tile_id(Tref, Pt[t + 1])
                        dx0, _ = Tref.seam_dist(Pt[t])
                        dx1, _ = Tref.seam_dist(Pt[t + 1])
                        crx = ke0 != ke1
                        ctx = (~crx) & (dx0 < 4) & (dx1 < 4)
                        jx.append(jump[crx])
                        jc.append(jump[ctx])
                        ae_x.append(Rt[t + 1].norm(dim=-1)[crx])
                        ae_c.append(Rt[t + 1].norm(dim=-1)[ctx])
                cat = lambda L: torch.cat(L) if L else torch.zeros(0, device=dev)
                a_cr, a_ct = cat(dcr), cat(dct)
                rec = {"fam": fam, "seed": seed, "rep": rep, "cfg": cname, "run": rn,
                       "n_cross": int(a_cr.numel()), "n_ctrl": int(a_ct.numel()),
                       "de_cross": sc(a_cr.mean()) if a_cr.numel() else None, "de_ctrl": sc(a_ct.mean()) if a_ct.numel() else None,
                       "absde_cross": sc(a_cr.abs().mean()) if a_cr.numel() else None, "absde_ctrl": sc(a_ct.abs().mean()) if a_ct.numel() else None,
                       "E": {str(t): E[t] - E[0] for t in (10, 20, 40)}, "ke0": float(0.5 * (V[0].double() ** 2).sum())}
                if rn != "exact":
                    J1, J2, A1, A2 = cat(jx), cat(jc), cat(ae_x), cat(ae_c)
                    rec.update({"jump_cross": sc(J1.mean()) if J1.numel() else None, "jump_ctrl": sc(J2.mean()) if J2.numel() else None,
                                "err_cross": sc(A1.mean()) if A1.numel() else None, "err_ctrl": sc(A2.mean()) if A2.numel() else None,
                                "n_jx": int(J1.numel()), "n_jc": int(J2.numel())})
                rows.append(rec)
        print("c done", fam, seed, rep, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kernel", default="old")
    ap.add_argument("--phase", choices=["a", "b", "c"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(4738)
    model = load(a.ckpt, dev)
    if a.kernel != "old":
        from scripts import scatter_far
        scatter_far.install_spec(model, a.kernel)
    rows = {"a": phase_a, "b": phase_b, "c": phase_c}[a.phase](model, dev, a)
    json.dump({"kernel": a.kernel, "rows": rows}, open(a.out, "w"))
    print("DONE", a.phase, len(rows), flush=True)


if __name__ == "__main__":
    main()
