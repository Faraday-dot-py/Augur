"""Contact-energy rows for the scatter-field regression harness (decision D3, option 3, 2026-10-13).

Scatter_regress.py's conservation rows (nr100, bh300 dE@1000 etc.) are the geometry-free GATE: gravity-only scenes, no walls, for contact checkpoints
with the learned gravity vector and ball-ball contact head zeroed (--contact --contact-free). These rows instead score contact scenes (walls, posts)
against the TRUE simulator's energy curve E_true(t) (contact_truth), never E(0): a model that dissipates like the truth passes, one that merely
conserves does not. Energy is read from the rolled trajectory (E = mean over balls of 0.5|v|^2 - g x, g = 9 along +x, the contact world's potential), so
nothing here assumes a gravity-only force formula inside the model. Counted only after the gate passes (scatter_regress.py compare).

Rows (seed-averaged; families box15 box_var post_circle post_rect disc post_poly, eval seeds 9100/9200/9300 x 24 scenes, 100 ticks):
  e_track@20/@100   median over scenes of |E_m(t) - E_t(t)| / max(|E_t(t)|, K0), K0 = true mean KE per ball at t=0; mean over families
  e_floor@20/@100   same for the truth re-run from positions perturbed by 1e-5 (chaos floor of the metric; reported, not gated)
  rest_dev          median over speeds of |e_model / e_truth - 1|, e = sqrt(rebound height / drop height) (apex after first reversal); wall v=4..15, circle v=4..10
  rest_miss         fraction of those speeds with no model rebound
  gain              max over speeds of (max_t E_m - max_t E_t) / (g * drop height) on the same single-ball drops (energy gained on a bounce beyond truth)
  runaway           max over families of fraction of scenes with any ball |dv|/dt > 200 after tick 20
  offgrid           max over families of fraction of balls outside the grid at tick 100
"""
import math

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_scenes as ss

G = ss.BOX_G
DT = cs.DT
FAMS = cs.TRAIN_FAMS + cs.HOLD_FAMS
STEPS = 100
TRACK_TICKS = (5, 20, 50, 100)
WALL_SPEEDS = (4, 6, 8, 10, 12, 15)
CIRCLE_SPEEDS = (4, 6, 8, 10)
THRESH = {"e_track@20": 0.2, "e_track@100": 0.7, "rest_dev": 0.25, "rest_miss": 0.2, "gain": 0.1, "runaway": 0.02, "offgrid": 0.02}


def energy(P, V, mask):
    """P, V (S,T,N,2), mask (S,N) -> per-scene mean per-ball E (S,T), and kinetic part."""
    cnt = mask.sum(1)[:, None]
    ke = (0.5 * (V ** 2).sum(-1) * mask[:, None]).sum(2) / cnt
    pe = (-G * P[..., 0] * mask[:, None]).sum(2) / cnt
    return ke + pe, ke


def track_err(Em, Et, K0):
    """Per-scene |E_m - E_t| / max(|E_t|, K0) (S,T); non-finite model energy counts as 1e6."""
    e = np.abs(Em - Et) / np.maximum(np.abs(Et), K0[:, None])
    return np.where(np.isfinite(e), e, 1e6)


def family_rows(model, fam, dev, extent, steps=STEPS):
    seeds = []
    for seed in cs.EVAL_SEEDS:
        es = cs.evalset(fam, seed, steps, dev)
        scenes, Pt, Vt = es["scenes"], es["Pt"], es["Vt"]
        Pm, Vm, mask = cs.roll(model, scenes, steps, dev)
        Em, _ = energy(Pm, Vm, mask)
        Et, Kt = energy(Pt, Vt, mask)
        K0 = Kt[:, 0]
        rngp = np.random.default_rng(seed + 1)
        pert = [cs.truth(s, steps, dev, ss.perturb(s, 1e-5, rngp)["pos"]) for s in scenes]
        n = Pt.shape[2]
        Ef, _ = energy(ss.pad_traj([p[0] for p in pert], n), ss.pad_traj([p[1] for p in pert], n), mask)
        em, ef = track_err(Em, Et, K0), track_err(Ef, Et, K0)
        acc = np.linalg.norm(np.diff(Vm, axis=1), axis=-1) / DT * mask[:, None]
        r = {"e_track": {str(k): float(np.median(em[:, k])) for k in TRACK_TICKS}, "e_floor": {str(k): float(np.median(ef[:, k])) for k in TRACK_TICKS},
             "E_model": {str(k): float(np.median(Em[:, k])) for k in TRACK_TICKS}, "E_true": {str(k): float(np.median(Et[:, k])) for k in TRACK_TICKS},
             "runaway": float((acc[:, 20:] > 200).any(axis=(1, 2)).mean()),
             "offgrid": float(((np.abs(Pm[:, steps]) > extent / 2).any(-1) * mask).sum() / mask.sum())}
        seeds.append(r)
    out = {}
    for k in ("e_track", "e_floor", "E_model", "E_true"):
        out[k] = {t: float(np.mean([s[k][t] for s in seeds])) for t in seeds[0][k]}
    for k in ("runaway", "offgrid"):
        out[k] = float(np.mean([s[k] for s in seeds]))
    print(f"contact-energy {fam} e_track {out['e_track']} floor {out['e_floor']} runaway {out['runaway']:.3f} offgrid {out['offgrid']:.3f}", flush=True)
    return out


def apex_ratio(x, wall, x0):
    """sqrt(rebound height / drop height) from the apex after the first velocity reversal; None if the ball never reverses."""
    vx = np.diff(x)
    rev = np.nonzero((vx[:-1] > 0) & (vx[1:] <= 0))[0]
    if not len(rev) or wall - x[rev[0] + 1:].min() <= 0:
        return None
    return float(math.sqrt((wall - x[rev[0] + 1:].min()) / max(wall - x0, 1e-9)))


def bounce_rows(model, dev, prims, face, speeds, steps=80):
    """Single ball dropped from rest toward a face at x=face (gravity +x), impact speed v = sqrt(2 g h). Returns per speed e_model, e_truth, gain."""
    out = {}
    wall = face - cs.R0
    for v in speeds:
        x0 = wall - v * v / (2 * G)
        if x0 < -6.25:
            continue
        s = dict(pos=np.array([[x0, 0.0]]), vel=np.zeros((1, 2)), mass=np.ones(1), rad=np.array([cs.R0]), prims=prims)
        Pt, Vt = cs.truth(s, steps, dev)
        Pm, Vm, mask = cs.roll(model, [s], steps, dev)
        Et, _ = energy(Pt[None], Vt[None], np.ones((1, 1)))
        Em, _ = energy(Pm, Vm, mask)
        gain = float((Em[0].max() if np.isfinite(Em[0]).all() else np.inf) - Et[0].max()) / (G * (wall - x0))
        out[str(v)] = {"e_model": apex_ratio(Pm[0, :, 0, 0], wall, x0), "e_truth": apex_ratio(Pt[:, 0, 0], wall, x0), "gain": gain}
    return out


def rest_summary(rows):
    dev_, miss, n = [], 0, 0
    for r in rows.values():
        if r["e_truth"] is None:
            continue
        n += 1
        if r["e_model"] is None:
            miss += 1
        else:
            dev_.append(abs(r["e_model"] / r["e_truth"] - 1))
    return (float(np.median(dev_)) if dev_ else float("inf")), miss / max(n, 1)


def summarize(res):
    fam = res["families"]
    mean = lambda k, t: float(np.mean([fam[f][k][t] for f in fam]))
    rows = {**res["restitution_wall"], **{"c" + k: v for k, v in res["restitution_circle"].items()}}
    wd, wm = rest_summary(res["restitution_wall"])
    cd, cm = rest_summary(res["restitution_circle"])
    return {"e_track@20": mean("e_track", "20"), "e_track@100": mean("e_track", "100"), "e_floor@20": mean("e_floor", "20"), "e_floor@100": mean("e_floor", "100"),
            "rest_dev": max(wd, cd), "rest_dev_wall": wd, "rest_dev_circle": cd, "rest_miss": max(wm, cm),
            "gain": max(r["gain"] for r in rows.values()), "runaway": max(f["runaway"] for f in fam.values()), "offgrid": max(f["offgrid"] for f in fam.values())}


def run_contact_energy(model, dev, extent, fams=FAMS):
    """model: ContactScatterField (set_scene) or plain ScatterField (zero-shot control, no geometry), built at dt = cs.DT."""
    box = [sc.hbox((-7, -7), (7, 7))]
    res = {"families": {f: family_rows(model, f, dev, extent) for f in fams}, "thresholds": THRESH}
    res["restitution_wall"] = bounce_rows(model, dev, box, 7.0, WALL_SPEEDS)
    res["restitution_circle"] = bounce_rows(model, dev, box + [sc.circle((2.0, 0.0), 2.0)], 0.0, CIRCLE_SPEEDS)
    res["summary"] = summarize(res)
    if hasattr(model, "set_scene"):
        model.set_scene(None, None)
    print("contact-energy summary", {k: round(v, 4) for k, v in res["summary"].items()}, flush=True)
    return res


def score(summary):
    """[(name, value, threshold, PASS/FAIL)] in THRESH order; value <= threshold passes (non-finite fails)."""
    return [(k, summary[k], t, "PASS" if summary[k] <= t else "FAIL") for k, t in THRESH.items()]
