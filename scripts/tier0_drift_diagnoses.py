"""Frozen-model rollout dump of the tiled closed-loop tier 0 (e44b796 checkpoint) and three no-training diagnoses from it.
  virial-and-loop-work-decomposition: virial ratio 2KE/|W| (densest third, per clump), closed-loop work of the learned force on rings of radius 1, 2, 4 around
    clump centres at visited states (massless probes in the free particle slots; exact-force control), drag/gain/radial regression of the force error at visited states
  restart-from-truth-drift-rate: d(dPE)/dtick of the densest third over 20 ticks from start ticks 28 and 60: continuing, truth state + 8-tick hidden warm-up,
    model state + hidden reset and 8-tick warm-up, truth state + the rollout's own hidden field
  integrator-compensation-test: same weights with two substeps of dt/2 (hidden field updated per substep, or once per tick); exact force at dt 0.1, 0.05, 0.025
Scenes: 24 per seed, seeds 9100/9200/9300, 100 ticks, same generator as scripts/tier0_close_pair_localise.py.
Usage: PYTHONPATH=. python scripts/tier0_drift_diagnoses.py --new <ckpt> --out <json> [--dump-dir <dir>]"""
import argparse
import json
import os

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import tier0_tiled_closed_loop as tc
from scripts import train_tier0_ext as tte
from scripts import train_tiered as ttr

SEEDS = (9100, 9200, 9300)
DT, EPS = ttr.DT, ttr.EPS
RADII = (1.0, 2.0, 4.0)
PROBE_TICKS = (28, 30, 60, 90)
STARTS = (28, 60)
WARM, WINDOW = 8, 20


def pair_r(pos, mask):
    pos = pos.double()
    d = pos[:, None] - pos[:, :, None]
    r = (d ** 2).sum(-1).sqrt()
    ok = (mask[:, :, None] * mask[:, None, :]).double() * (1 - torch.eye(pos.shape[1], device=pos.device, dtype=torch.float64))
    return r, ok


def pe_particle(pos, mask):
    r, ok = pair_r(pos, mask)
    return -0.5 * (ok / torch.sqrt(r ** 2 + EPS ** 2)).sum(-1)


def densest_third(pos20, mask):
    r, ok = pair_r(pos20, mask)
    cnt = ((r < 4.0).double() * ok).sum(-1)
    sel = torch.zeros_like(mask, dtype=torch.bool)
    for s in range(len(mask)):
        c = cnt[s][mask[s] > 0]
        sel[s] = (cnt[s] > torch.quantile(c, 2 / 3)) & (mask[s] > 0)
    return sel


def energies(P, V, M, mask):
    return sf.energy(P.double(), V.double(), M.double(), mask.double(), EPS)


def ring_work(F, ring):
    """Closed-loop work of F (.., K, 2) along the closed polyline ring (.., K, 2), and the closed-loop integral of |F||dl|."""
    dx = torch.roll(ring, -1, -2) - ring
    Fm = 0.5 * (F + torch.roll(F, -1, -2))
    w = (Fm * dx).sum((-1, -2))
    nrm = (0.5 * (F.norm(dim=-1) + torch.roll(F.norm(dim=-1), -1, -1)) * dx.norm(dim=-1)).sum(-1)
    return w, nrm


def regress_force_error(e, v, at, u):
    """Least squares e = c0 * (-v) + c1 * at + c2 * u over stacked components; e, v, at, u (n, 2). Returns coefficients, R^2 and the share of the signed
    work error sum(v . e) carried by each term."""
    y = e.reshape(-1)
    X = torch.stack([(-v).reshape(-1), at.reshape(-1), u.reshape(-1)], 1)
    c = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    r2 = 1.0 - float(((y - X @ c) ** 2).sum() / (y ** 2).sum().clamp_min(1e-30))
    vf = v.reshape(-1)
    work = float((vf * y).sum())
    contrib = [float(c[j] * (vf * X[:, j]).sum()) for j in range(3)]
    return {"coef_drag": float(c[0]), "coef_gain": float(c[1]), "coef_inward": float(c[2]), "r2": r2, "work_error_sum": work,
            "work_share_drag": contrib[0] / work if abs(work) > 1e-30 else float("nan"),
            "work_share_gain": contrib[1] / work if abs(work) > 1e-30 else float("nan"),
            "work_share_inward": contrib[2] / work if abs(work) > 1e-30 else float("nan")}


def slope(y):
    """Least squares slope of y (T,) per tick."""
    t = torch.arange(len(y), dtype=torch.float64)
    t = t - t.mean()
    y = torch.as_tensor(y, dtype=torch.float64)
    return float((t * (y - y.mean())).sum() / (t ** 2).sum())


def centres_by_label(P, lab, mask):
    """(S, 4, 2) mean position per cluster label and (S, 4) existence."""
    cs, ex = [], []
    for c in range(4):
        w = ((lab == c) & (mask > 0)).double()
        n = w.sum(-1)
        cs.append((P.double() * w[..., None]).sum(1) / n.clamp_min(1)[:, None])
        ex.append(n > 1)
    return torch.stack(cs, 1), torch.stack(ex, 1)


def probe_force(fn, pos, vel, mass, mask, fld, ring):
    """Force of fn at massless probes. ring (S, K, 2) is put into the first K free slots of every scene (mass 0, velocity 0)."""
    S, K = ring.shape[:2]
    pos2, vel2, mass2, mask2 = pos.clone(), vel.clone(), mass.clone(), mask.clone()
    idx = torch.stack([(mask[s] == 0).nonzero()[:K, 0] for s in range(S)])
    ar = torch.arange(S, device=pos.device)[:, None]
    pos2[ar, idx], vel2[ar, idx], mass2[ar, idx], mask2[ar, idx] = ring.to(pos.dtype), 0.0, 0.0, 1.0
    a = fn(pos2, vel2, mass2, mask2, fld)
    return a[ar, idx], ttr.true_acc(pos2, mass2, mask2)[ar, idx]


def loop_work_at(tt, P, V, M, mk, fld, cen, ex, npts=64):
    th = torch.arange(npts, device=P.device, dtype=torch.float64) * (2 * np.pi / npts)
    unit = torch.stack([th.cos(), th.sin()], -1)
    out = {}
    for R in RADII:
        mod, ctl = [], []
        for c in range(4):
            ring = cen[:, c, None, :] + R * unit[None]
            a, at = probe_force(lambda p, v, m, k, f: tt.force(p, v, m, k, f)[0], P, V, M, mk, fld, ring)
            for lst, F in ((mod, a), (ctl, at)):
                w, n = ring_work(F.double(), ring)
                lst.append(torch.where(ex[:, c], (w / n.clamp_min(1e-12)).abs(), torch.full_like(w, float("nan"))))
        out[str(R)] = {"model": torch.stack(mod, 1), "exact": torch.stack(ctl, 1)}
    return out


@torch.no_grad()
def roll_dump(tt, d, T, probe_ticks):
    m, mk = d["M"].float(), d["mask"].float()
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    fld = tt.init_state(len(pos))
    a_prev, fld = tt.force(pos, vel, m, mk, fld)
    P, V, A, flds = [pos], [vel], [a_prev], {}
    for t in range(1, T + 1):
        pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
        a_new, fld = tt.force(pn, vel + DT * a_prev, m, mk, fld)
        pos, vel, a_prev = pn, vel + 0.5 * DT * (a_prev + a_new), a_new
        P.append(pos)
        V.append(vel)
        A.append(a_new)
        if t in probe_ticks:
            flds[t] = fld.clone()
    return torch.stack(P, 1), torch.stack(V, 1), torch.stack(A, 1), flds


@torch.no_grad()
def run_from(tt, d, pos, vel, a_prev, fld, T):
    m, mk = d["M"].float(), d["mask"].float()
    P = [pos]
    for _ in range(T):
        pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
        a_new, fld = tt.force(pn, vel + DT * a_prev, m, mk, fld)
        pos, vel, a_prev = pn, vel + 0.5 * DT * (a_prev + a_new), a_new
        P.append(pos)
    return torch.stack(P, 1)


@torch.no_grad()
def warm_start(tt, d, Pw, Vw):
    """Fresh hidden field teacher-forced over the states Pw, Vw (S, W+1, N, 2); returns the force and hidden field at the last state."""
    m, mk = d["M"].float(), d["mask"].float()
    fld = tt.init_state(len(m))
    for i in range(Pw.shape[1]):
        a, fld = tt.force(Pw[:, i].float(), Vw[:, i].float(), m, mk, fld)
    return a, fld


@torch.no_grad()
def substep_roll(tt, d, T, hidden_each_substep):
    m, mk = d["M"].float(), d["mask"].float()
    dt = DT / 2
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    fld = tt.init_state(len(pos))
    a_prev, fld = tt.force(pos, vel, m, mk, fld)
    P, V = [pos], [vel]
    for t in range(1, T + 1):
        for sub in range(2):
            pn = pos + dt * vel + 0.5 * dt ** 2 * a_prev
            a_new, f2 = tt.force(pn, vel + dt * a_prev, m, mk, fld)
            if hidden_each_substep or sub == 1:
                fld = f2
            pos, vel, a_prev = pn, vel + 0.5 * dt * (a_prev + a_new), a_new
        P.append(pos)
        V.append(vel)
    return torch.stack(P, 1), torch.stack(V, 1)


@torch.no_grad()
def oracle_roll(d, T, dt):
    m, mk = d["M"].float(), d["mask"].float()
    n = int(round(DT / dt))
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    a_prev = ttr.true_acc(pos, m, mk)
    P, V = [pos], [vel]
    for t in range(1, T + 1):
        for _ in range(n):
            pn = pos + dt * vel + 0.5 * dt ** 2 * a_prev
            a_new = ttr.true_acc(pn, m, mk)
            pos, vel, a_prev = pn, vel + 0.5 * dt * (a_prev + a_new), a_new
        P.append(pos)
        V.append(vel)
    return torch.stack(P, 1), torch.stack(V, 1)


def virial(P, V, mask, sel=None):
    """Per scene 2KE/|PE| over sel particles (all if None), PE as the sum of per-particle half-pair potentials."""
    w = mask.double() if sel is None else sel.double()
    ke = 0.5 * ((V.double() ** 2).sum(-1) * w).sum(-1)
    pe = (pe_particle(P, mask) * w).sum(-1)
    return 2 * ke / pe.abs().clamp_min(1e-12)


def clump_virial(P, V, mask, lab):
    out = []
    for c in range(4):
        w = ((lab == c) & (mask > 0)).double()
        n = w.sum(-1)
        vc = (V.double() * w[..., None]).sum(1) / n.clamp_min(1)[:, None]
        ke = 0.5 * (((V.double() - vc[:, None]) ** 2).sum(-1) * w).sum(-1)
        r, ok = pair_r(P, mask)
        ok = ok * w[:, :, None] * w[:, None, :]
        pe = -0.5 * (ok / torch.sqrt(r ** 2 + EPS ** 2)).sum((-1, -2))
        out.append(torch.where(n > 1, 2 * ke / pe.abs().clamp_min(1e-12), torch.full_like(ke, float("nan"))))
    return torch.stack(out, 1)


def nanmean(x):
    return float(torch.nanmean(x.double()))


def dpe_series(P, Pt, mask, sel, e0):
    """(T+1, S) selected-particle potential energy, model minus truth, over |E_truth(0)|."""
    return torch.stack([((pe_particle(P[:, t], mask) - pe_particle(Pt[:, t], mask)) * sel).sum(-1) / e0 for t in range(P.shape[1])], 0)


def scene_diagnoses(tt, d, sd, dump_dir):
    dev = d["P"].device
    S = len(d["M"])
    Pt, Vt, M, mk, lab = d["P"], d["V"], d["M"], d["mask"], d["lab"]
    e0 = energies(Pt[:, 0], Vt[:, 0], M, mk).abs()
    sel = densest_third(Pt[:, 20], mk)
    P, V, A, flds = roll_dump(tt, d, 100, PROBE_TICKS)
    if dump_dir:
        torch.save({"P": P.cpu(), "V": V.cpu(), "A": A.cpu(), "flds": {k: v.cpu() for k, v in flds.items()}}, os.path.join(dump_dir, f"rollout_dump_seed{sd}.pt"))
    res = {"E0abs_mean": float(e0.mean()), "n_particles_mean": float(mk.sum(-1).mean()), "n_dense_third_mean": float(sel.sum(-1).double().mean())}
    Pm, Vm = P.double(), V.double()
    # rollout headline numbers
    res["rollout"] = {}
    for k in (20, 50, 100):
        em = energies(Pm[:, k], Vm[:, k], M, mk)
        et0 = energies(Pt[:, 0], Vt[:, 0], M, mk)
        res["rollout"][str(k)] = {"dE_vs_E0_abs_mean": float(((em - et0).abs() / e0).mean()), "dE_vs_E0_signed_mean": float(((em - et0) / e0).mean()),
                                  "dKE_vs_truth_t": float(((0.5 * (Vm[:, k] ** 2).sum(-1) * mk).sum(-1) - (0.5 * (Vt[:, k] ** 2).sum(-1) * mk).sum(-1)).div(e0).mean()),
                                  "dPE_vs_truth_t": float(((pe_particle(Pm[:, k], mk) * mk).sum(-1) - (pe_particle(Pt[:, k], mk) * mk).sum(-1)).div(e0).mean()),
                                  "dPE_dense_third_vs_truth_t": float(((pe_particle(Pm[:, k], mk) - pe_particle(Pt[:, k], mk)) * sel).sum(-1).div(e0).mean()),
                                  "err_per_body": float((((Pm[:, k] - Pt[:, k]) ** 2).sum(-1).sqrt() * mk).sum() / mk.sum())}
    # diagnosis 1: virial
    vir = {}
    for k in (20, 40, 60, 100):
        vm, vt = virial(Pm[:, k], Vm[:, k], mk, sel), virial(Pt[:, k], Vt[:, k], mk, sel)
        am, at_ = virial(Pm[:, k], Vm[:, k], mk), virial(Pt[:, k], Vt[:, k], mk)
        cm, ct = clump_virial(Pm[:, k], Vm[:, k], mk, lab), clump_virial(Pt[:, k], Vt[:, k], mk, lab)
        vir[str(k)] = {"dense_third_model": float(vm.mean()), "dense_third_truth": float(vt.mean()), "dense_third_rel_diff": float(((vm - vt) / vt).mean()),
                       "all_model": float(am.mean()), "all_truth": float(at_.mean()), "all_rel_diff": float(((am - at_) / at_).mean()),
                       "per_clump_model": nanmean(cm), "per_clump_truth": nanmean(ct), "per_clump_rel_diff": nanmean((cm - ct) / ct)}
    res["virial"] = vir
    # diagnosis 1: loop work
    loops = {}
    for k in (30, 60, 90):
        cen, exk = centres_by_label(Pm[:, k], lab, mk)
        lw = loop_work_at(tt, P[:, k].float(), V[:, k].float(), M.float(), mk.float(), flds[k], cen, exk)
        loops[str(k)] = {R: {"model_mean_abs_ratio": nanmean(v["model"]), "exact_control_mean_abs_ratio": nanmean(v["exact"])} for R, v in lw.items()}
    res["loop_work"] = loops
    # diagnosis 1: drag regression at visited states
    reg = {}
    for name, (lo, hi) in (("ticks30-60", (30, 60)), ("ticks60-100", (60, 100))):
        for subset in ("dense_third", "all"):
            es, vs, ats, us = [], [], [], []
            for t in range(lo, hi + 1):
                cen, exk = centres_by_label(Pm[:, t], lab, mk)
                lc = lab.clamp_min(0)
                ctr = torch.gather(cen, 1, lc[..., None].expand(S, lab.shape[1], 2))
                u = ctr - Pm[:, t]
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                w = (sel if subset == "dense_third" else mk > 0)
                at = ttr.true_acc(P[:, t].float(), M.float(), mk.float()).double()
                es.append((A[:, t].double() - at)[w])
                vs.append(Vm[:, t][w])
                ats.append(at[w])
                us.append(u[w])
            r = regress_force_error(torch.cat(es), torch.cat(vs), torch.cat(ats), torch.cat(us))
            reg[f"{name}/{subset}"] = r
    res["drag_regression"] = reg
    # diagnosis 2: restart
    rest = {}
    for s0 in STARTS:
        lo, hi = s0 - WARM, s0
        series = {"continuing": dpe_series(Pm[:, s0:s0 + WINDOW + 1], Pt[:, s0:s0 + WINDOW + 1], mk, sel, e0)}
        a_t, f_t = warm_start(tt, d, Pt[:, lo:hi + 1], Vt[:, lo:hi + 1])
        Pr = run_from(tt, d, Pt[:, s0].float(), Vt[:, s0].float(), a_t, f_t, WINDOW)
        series["truth_state_warm_hidden"] = dpe_series(Pr, Pt[:, s0:s0 + WINDOW + 1], mk, sel, e0)
        a_m, f_m = warm_start(tt, d, Pm[:, lo:hi + 1].float(), Vm[:, lo:hi + 1].float())
        Pr = run_from(tt, d, P[:, s0], V[:, s0], a_m, f_m, WINDOW)
        series["model_state_reset_hidden"] = dpe_series(Pr, Pt[:, s0:s0 + WINDOW + 1], mk, sel, e0)
        a_x, _ = tt.force(Pt[:, s0].float(), Vt[:, s0].float(), M.float(), mk.float(), flds[s0])
        Pr = run_from(tt, d, Pt[:, s0].float(), Vt[:, s0].float(), a_x, flds[s0], WINDOW)
        series["truth_state_rollout_hidden"] = dpe_series(Pr, Pt[:, s0:s0 + WINDOW + 1], mk, sel, e0)
        rest[str(s0)] = {k: {"slope_per_tick_mean": float(torch.stack([torch.tensor(slope(v[:, i].cpu())) for i in range(S)]).mean()),
                             "dPE_start": float(v[0].mean()), "dPE_end": float(v[-1].mean())} for k, v in series.items()}
    res["restart"] = rest
    # diagnosis 4: substeps and oracle
    sub = {}
    et0 = energies(Pt[:, 0], Vt[:, 0], M, mk)
    def dsum(Pq, Vq):
        em = energies(Pq[:, 100], Vq[:, 100], M, mk)
        return {"dE_vs_E0_abs_mean": float(((em - et0).abs() / e0).mean()), "dE_vs_E0_signed_mean": float(((em - et0) / e0).mean()),
                "dPE_dense_third_vs_truth_t": float(((pe_particle(Pq[:, 100].double(), mk) - pe_particle(Pt[:, 100], mk)) * sel).sum(-1).div(e0).mean()),
                "dPE_all_vs_truth_t": float(((pe_particle(Pq[:, 100].double(), mk) - pe_particle(Pt[:, 100], mk)) * mk).sum(-1).div(e0).mean()),
                "err_per_body_100": float((((Pq[:, 100].double() - Pt[:, 100]) ** 2).sum(-1).sqrt() * mk).sum() / mk.sum())}
    sub["model_dt0.1"] = dsum(P, V)
    for name, each in (("model_dt0.05_hidden_per_substep", True), ("model_dt0.05_hidden_per_tick", False)):
        Pq, Vq = substep_roll(tt, d, 100, each)
        sub[name] = dsum(Pq, Vq)
    for dt in (0.1, 0.05, 0.025):
        Pq, Vq = oracle_roll(d, 100, dt)
        sub[f"oracle_dt{dt}"] = dsum(Pq, Vq)
    res["integrator"] = sub
    return res


def pool(per_seed):
    """Mean over seeds of every numeric leaf."""
    first = per_seed[0]
    if isinstance(first, dict):
        return {k: pool([p[k] for p in per_seed]) for k in first}
    if isinstance(first, (int, float)) and first is not None:
        return float(np.nanmean([p for p in per_seed]))
    return None


def verdicts(pooled):
    v = {}
    vir = pooled["virial"]["100"]["dense_third_rel_diff"]
    loop = float(np.mean([pooled["loop_work"][k][R]["model_mean_abs_ratio"] for k in pooled["loop_work"] for R in pooled["loop_work"][k]]))
    ctl = float(np.mean([pooled["loop_work"][k][R]["exact_control_mean_abs_ratio"] for k in pooled["loop_work"] for R in pooled["loop_work"][k]]))
    drag = pooled["drag_regression"]["ticks60-100/dense_third"]["work_share_drag"]
    ok = (vir <= -0.10) and (loop >= 0.03 or drag >= 0.40)
    kill = loop < 0.005 and drag < 0.10
    v["virial-and-loop-work-decomposition"] = {"virial_rel_diff_tick100_dense_third": vir, "loop_work_ratio_model": loop, "loop_work_ratio_exact_control": ctl,
                                               "drag_work_share_ticks60-100_dense_third": drag, "verdict": "PASS" if ok else ("KILL" if kill else "INCONCLUSIVE")}
    r = {}
    for s0 in pooled["restart"]:
        x = pooled["restart"][s0]
        r[s0] = {k: x[k]["slope_per_tick_mean"] for k in x}
    a, b, c, h = (r["60"][k] for k in ("continuing", "truth_state_warm_hidden", "model_state_reset_hidden", "truth_state_rollout_hidden"))
    neg = lambda z: -z
    if neg(a) <= 0:
        cause = "NO_NEGATIVE_RATE_AT_START_60"
    elif neg(a) > 0 and abs(neg(a) - neg(c)) <= 0.3 * neg(a) and neg(a) >= 2 * neg(b):
        cause = "visited-state"
    elif neg(b) >= 0.7 * neg(a):
        cause = "per-tick-bias"
    elif neg(c) < 0.5 * neg(a):
        cause = "hidden-state"
    else:
        cause = "mixed"
    v["restart-from-truth-drift-rate"] = {"slopes_by_start": r, "cause_by_start60": cause}
    sub = pooled["integrator"]
    base = sub["model_dt0.1"]["dE_vs_E0_abs_mean"]
    chg = {k: abs(sub[k]["dE_vs_E0_abs_mean"] - base) / base for k in ("model_dt0.05_hidden_per_substep", "model_dt0.05_hidden_per_tick")}
    prim = chg["model_dt0.05_hidden_per_tick"]
    v["integrator-compensation-test"] = {"relative_change_dE100_by_variant": chg, "primary_variant": "model_dt0.05_hidden_per_tick",
                                         "oracle_dPE_dense_third_dt0.1": sub["oracle_dt0.1"]["dPE_dense_third_vs_truth_t"],
                                         "oracle_dPE_dense_third_dt0.05": sub["oracle_dt0.05"]["dPE_dense_third_vs_truth_t"],
                                         "verdict": "PASS" if prim >= 0.30 else ("KILL" if prim < 0.10 else "INCONCLUSIVE")}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = tte.build(argparse.Namespace(t0="ext=6,extapply,taper=8,bnd,halo=8", dt=0.1), dev)
    sf.load_compat(m, torch.load(a.new, map_location=dev, weights_only=False)["model"])
    m.eval().requires_grad_(False)
    tt = tc.TiledTier0(m, dev, chunk=a.chunk, exchange="halo", ckpt=False)
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
    per = {}
    for sd in SEEDS:
        d = ttr.make_scenes(a.scenes, 100, sd, dev)
        per[str(sd)] = scene_diagnoses(tt, d, sd, a.dump_dir)
        json.dump({"per_seed": per}, open(a.out + ".partial", "w"), indent=1, default=lambda o: None)
        print(sd, "done", flush=True)
    pooled = pool([per[str(sd)] for sd in SEEDS])
    out = {"per_seed": per, "pooled": pooled, "verdicts": verdicts(pooled)}
    json.dump(out, open(a.out, "w"), indent=1, default=lambda o: None)
    print(json.dumps(out["verdicts"], indent=1), flush=True)


if __name__ == "__main__":
    main()
