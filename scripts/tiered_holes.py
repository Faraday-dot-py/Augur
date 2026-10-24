"""Evaluation drivers for the remaining arms of docs/theory/heterogeneous-tiered.md, ready for models as they appear.
Each driver returns a JSON dict with a verdict from a pre-registered rule (the rule text is in the driver docstring; rules the doc states
verbatim are marked DOC, the rest are derived from the doc's wording and fixed here before any run). Seeds 4738 then 9100/9200/9300; verdicts
use the held-out seeds, paired, with threshold = 2 standard errors of the paired difference over those seeds.

  respa      hole 10: multi-rate (RESPA, TieredAdapter.step_multirate) vs synchronous
  exposure   hole 5: arms S (teacher-forced), R (rollout-in-loop), N (noise injection), J (joint) vs S; gap diagnostic
  sparse     hole 8: sparse tile activation vs dense, cost vs occupancy, optional refinement-tier model
  recurrent  hole 7: recurrent-field halo handling (reset / persist / refresh) and shift invariance
  calibration error head: rank correlation, decile reliability, out-of-distribution rise

  python scripts/tiered_holes.py respa --model tiered:<nets.pt> --out results/tiered/holes_respa.json
  python scripts/tiered_holes.py exposure --arms S=tiered:<S.pt> R=tiered:<R.pt> [N=... J=...] --out ...
  python scripts/tiered_holes.py sparse --model tiered:<nets.pt> [--refined tiered:<refined.pt>] --out ...
  python scripts/tiered_holes.py recurrent --model scatter:<E.pt> --out ...
  python scripts/tiered_holes.py calibration --model <spec with .error> --out ..."""
import argparse
import json
import math
import time

import numpy as np
import torch

from scripts import tiered_arms as ta
from scripts import tiered_eval as te
from scripts import tiered_iface as ti
from scripts import tiered_truth as tt

HELD = (9100, 9200, 9300)


def pair_se(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), (float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf"))


def better(a, b, lower_is_better=True):
    """(mean gain of a over b, se, significant): gain = b - a for lower-is-better metrics."""
    m, se = pair_se(b, a) if lower_is_better else pair_se(a, b)
    return m, se, bool(m > 2 * se)


def sync_sub(model, b):
    return max(1, int(round(model.dt / (b["dt"] / b["sub"]))))


def clock(fn, cuda):
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    out = fn()
    if cuda:
        torch.cuda.synchronize()
    return out, time.time() - t


# ------------------------------------------------------------------ hole 10

@torch.no_grad()
def roll_multirate(model, batch, T, m):
    pos, vel, mass, mask = batch["pos"], batch["vel"], batch["mass"], batch["mask"]
    st = model.init_state(pos.shape[0], pos.device)
    P, V = [pos], [vel]
    for _ in range(T // m):
        if m == 1:
            pos, vel, st = ti.advance(model, pos, vel, mass, mask, st, dict(phys=batch["phys"]))
        else:
            pos, vel, st = model.step_multirate(pos, vel, mass, mask, st, m)
        pos, vel = te.fin(pos), te.fin(vel)
        P.append(pos), V.append(vel)
    return dict(pos=torch.stack(P, 1), vel=torch.stack(V, 1))


def respa_verdict(rows):
    """rows: {m: {err:[per seed], dE:[...], ms:[...]}} with m=1 the synchronous reference.
    PASS (per m>1): err <= 1.10x sync (or within 2 SE), |dE| <= sync |dE| (or within 2 SE), and ms per physical step <= 0.85x sync.
    KILL (per m): err > 1.25x sync beyond 2 SE or |dE| > 2x sync. Otherwise MIXED. Overall: PASS if any m passes, KILL if all m kill."""
    out, ref = {}, rows[1]
    for m, r in rows.items():
        if m == 1:
            continue
        e_m, e_se = pair_se(r["err"], ref["err"])
        d_m, d_se = pair_se(np.abs(r["dE"]), np.abs(ref["dE"]))
        err_ok = np.mean(r["err"]) <= 1.10 * np.mean(ref["err"]) or e_m <= 2 * e_se
        dE_ok = d_m <= 0 or d_m <= 2 * d_se
        cost = float(np.mean(r["ms"]) / np.mean(ref["ms"]))
        kill = (np.mean(r["err"]) > 1.25 * np.mean(ref["err"]) and e_m > 2 * e_se) or np.mean(np.abs(r["dE"])) > 2 * np.mean(np.abs(ref["dE"])) + 1e-9
        out[m] = dict(err_ratio=float(np.mean(r["err"]) / np.mean(ref["err"])), dE_diff=d_m, cost_ratio=cost,
                      verdict="PASS" if (err_ok and dE_ok and cost <= 0.85) else ("KILL" if kill else "MIXED"))
    vs = [v["verdict"] for v in out.values()]
    return dict(per_m=out, verdict="PASS" if "PASS" in vs else ("KILL" if vs and all(v == "KILL" for v in vs) else "MIXED"))


def run_respa(model, dev, seeds, T=100, ms=(1, 2, 4)):
    if not hasattr(model, "step_multirate"):
        return dict(verdict="skipped: model has no step_multirate")
    rows = {m: dict(err=[], dE=[], ms=[]) for m in ms}
    for s in seeds:
        b = te.grav_batch("multiclump", 300, ta.SPAN, s, 4, dev)
        rt = te.truth_roll(b, model.dt, T)
        ct = te.conserved(rt, b, None, model.dt)
        for m in ms:
            r, secs = clock(lambda: roll_multirate(model, b, T, m), dev.type == "cuda")
            idx = torch.arange(0, T + 1, m, device=dev)
            e = te.err_curve(r["pos"], rt["pos"][:, idx], b["mask"])[:, -1].mean()
            rr = dict(pos=r["pos"], vel=r["vel"])
            ce = te.conserved(rr, b, None, model.dt * m)
            rows[m]["err"].append(float(e)), rows[m]["dE"].append(float(ce["dE"][:, -1].abs().mean())), rows[m]["ms"].append(secs / T * 1e3)
    return dict(rows=rows, **respa_verdict({m: dict(err=v["err"], dE=v["dE"], ms=v["ms"]) for m, v in rows.items()}))


# ------------------------------------------------------------------ hole 5

def exposure_arm(model, dev, seed, T=100):
    b = te.grav_batch("multiclump", 300, ta.SPAN, seed, 4, dev)
    rt, rm = te.truth_roll(b, model.dt, T), te.model_roll(model, b, T)
    ec = te.err_curve(rm["pos"], rt["pos"], b["mask"])
    cm, ct = te.conserved(rm, b, None, model.dt), te.conserved(rt, b, None, model.dt)
    gaps = {}
    for k in (10, 50, 100):
        if k > T:
            continue
        errs = {}
        for nm, tr in (("truth_states", rt), ("model_states", rm)):
            bb = dict(b, pos=tr["pos"][:, k], vel=tr["vel"][:, k])
            a_t = tt.grav_acc(bb["pos"], b["mass"], b["mask"])
            a_m = ta.settled_force(model, bb)
            errs[nm] = float(((a_m - a_t) ** 2).sum().sqrt() / (a_t ** 2).sum().sqrt())
        gaps[str(k)] = dict(errs, ratio=errs["model_states"] / max(errs["truth_states"], 1e-12))
    out = dict(err100=float(ec[:, min(T, 100)].mean()), err50=float(ec[:, min(T, 50)].mean()), dE100=float(cm["dE"][:, -1].abs().mean()),
               dE100_truth=float(ct["dE"][:, -1].abs().mean()), gap=gaps)
    t0 = model.tier0_only() if hasattr(model, "tier0_only") else None
    if t0 is not None:
        bias = te.g_bias(t0, dev, seed, False)
        out["tier0_bias_absz"] = float(max(abs(z) for z in bias["z"]))
    return out


def exposure_verdict(res):
    """res: {arm: {seed: exposure_arm output}} with arm S as reference. DOC (hole 5): gap within seed noise -> teacher forcing suffices;
    pass for an arm: improvement over S in rollout error (err100) and energy drift (dE100) beyond 2 SE, hole-4 bias not regressing
    (tier-0 max|z| not above max(S, 2) + 0.5); DOC kill: arm improves rollout error but fails the bias check."""
    seeds = [s for s in res["S"] if int(s) in HELD]
    g = lambda arm, key: [res[arm][s][key] for s in seeds]
    out = {}
    gap_s = [res["S"][s]["gap"]["50"]["ratio"] for s in seeds if "50" in res["S"][s]["gap"]]
    gm = float(np.mean(gap_s)) if gap_s else float("nan")
    gse = float(np.std(gap_s, ddof=1) / math.sqrt(len(gap_s))) if len(gap_s) > 1 else float("inf")
    out["S_gap_ratio_step50"] = dict(mean=gm, se=gse, teacher_forcing_suffices=bool(abs(gm - 1) <= 2 * gse))
    for arm in res:
        if arm == "S":
            continue
        e_m, e_se, e_sig = better(g(arm, "err100"), g("S", "err100"))
        d_m, d_se, d_sig = better(g(arm, "dE100"), g("S", "dE100"))
        zs, za = max(g("S", "tier0_bias_absz") or [0]), max(g(arm, "tier0_bias_absz") or [0])
        bias_ok = za <= max(zs, 2.0) + 0.5
        v = "PASS" if (e_sig and d_sig and bias_ok) else ("KILL" if (e_sig and not bias_ok) else ("MIXED" if (e_sig or d_sig) else "NO_GAIN"))
        out[arm] = dict(err100_gain=e_m, err100_se=e_se, dE100_gain=d_m, dE100_se=d_se, bias_absz=za, bias_absz_S=zs, verdict=v)
    return out


# ------------------------------------------------------------------ hole 8

def sparse_verdict(r):
    """PASS: sparse and dense forces agree (rel rms <= 1e-3) AND sparse time at 12.5% occupancy <= 0.6x dense time at full occupancy.
    Refinement tier (if given): PASS if cold-collapse err@50 improves >= 10% and beyond 2 SE with far-field error not worse than +5%.
    KILL: sparse disagrees with dense (> 1e-2) or refinement gives no gain (< 2 SE)."""
    out = dict(equiv_ok=r["equiv_rel_rms"] <= 1e-3, cost_ok=r["ms_sparse_0125"] <= 0.6 * r["ms_dense_1"])
    out["sparse"] = "PASS" if out["equiv_ok"] and out["cost_ok"] else ("KILL" if r["equiv_rel_rms"] > 1e-2 else "MIXED")
    if "refine" in r:
        f = r["refine"]
        gain = 1 - np.mean(f["err50_refined"]) / np.mean(f["err50_base"])
        m, se, sig = better(f["err50_refined"], f["err50_base"])
        far_ok = np.mean(f["far_refined"]) <= 1.05 * np.mean(f["far_base"])
        out["refine"] = dict(gain=float(gain), significant=sig, far_ok=bool(far_ok), verdict="PASS" if (gain >= 0.10 and sig and far_ok) else ("KILL" if not sig else "MIXED"))
    return out


@torch.no_grad()
def run_sparse(model, dev, seeds, refined=None):
    if not hasattr(model, "f"):
        return dict(verdict="skipped: needs TieredAdapter")
    f = model.f
    old = f.static
    b = te.grav_batch("multiclump", 300, ta.SPAN, seeds[0], 4, dev)
    out = {}
    f.static = True
    a_d = ta.settled_force(model, b)
    f.static = False
    a_s = ta.settled_force(model, b)
    out["equiv_rel_rms"] = float(((a_s - a_d) ** 2).sum().sqrt() / (a_d ** 2).sum().sqrt())
    rng = np.random.default_rng(seeds[0])
    for mode in ("dense", "sparse"):
        f.static = mode == "dense"
        for occ in (1.0, 0.5, 0.25, 0.125):
            n = 2000
            w = ta.SPAN / 2 * 0.95 * math.sqrt(occ)
            bb = dict(pos=torch.as_tensor(rng.uniform(-w, w, (1, n, 2)), device=dev), vel=torch.as_tensor(rng.normal(0, 0.3, (1, n, 2)), device=dev),
                      mass=torch.full((1, n), 1.0, device=dev, dtype=torch.float64), mask=torch.ones(1, n, device=dev, dtype=torch.float64), phys=b["phys"])
            t, mem = te.timed_step(model, bb)
            out[f"ms_{mode}_{str(occ).replace('.', '')}"] = t
            out[f"mb_{mode}_{str(occ).replace('.', '')}"] = mem
    out["ms_sparse_0125"], out["ms_dense_1"] = out["ms_sparse_0125"], out["ms_dense_10"]
    f.static = old
    if refined is not None:
        e = dict(err50_base=[], err50_refined=[], far_base=[], far_refined=[])
        for s in seeds:
            if s not in HELD:
                continue
            bc = te.grav_batch("cold_collapse", 300, 44.0, s, 4, dev)
            bc = dict(bc, pos=bc["pos"] * 2.0)
            rt = te.truth_roll(bc, model.dt, 50)
            for nm, mdl in (("base", model), ("refined", refined)):
                e[f"err50_{nm}"].append(float(te.err_curve(te.model_roll(mdl, bc, 50)["pos"], rt["pos"], bc["mask"])[:, -1].mean()))
                e[f"far_{nm}"].append(ta.far_metrics(mdl, s, dev)["rel_rms_far_dominated"])
        out["refine"] = e
    out["verdict"] = sparse_verdict(out)
    return out


# ------------------------------------------------------------------ hole 7

def recurrent_verdict(r):
    """Halo handling of the recurrent field (reset / persist / refresh). Derived from hole 7: PASS for 'refresh' if err@50 <= 0.90x persist beyond 2 SE
    or its seam jump (jump / field, g_seams) is <= 0.5x persist's; KILL if neither and cost > 1.2x. Shift invariance: PASS if error under a half-tile
    shift <= 2x the numerical floor (shift by 2 cells inside one tile; the half-tile shift of 16 units moves the cluster across the tile seam) + 1e-3 relative."""
    out = {}
    m, se, sig = better(r["refresh"]["err50"], r["persist"]["err50"])
    jump_ok = r["refresh"]["seam_jump"] <= 0.5 * r["persist"]["seam_jump"]
    cost = r["refresh"]["ms"] / r["persist"]["ms"]
    out["refresh"] = dict(err_gain=m, err_se=se, significant=sig, seam_jump_ratio=r["refresh"]["seam_jump"] / max(r["persist"]["seam_jump"], 1e-12), cost_ratio=cost,
                          verdict="PASS" if ((sig and np.mean(r["refresh"]["err50"]) <= 0.9 * np.mean(r["persist"]["err50"])) or jump_ok) else ("KILL" if cost > 1.2 else "MIXED"))
    out["persist_vs_reset"] = dict(zip(("gain", "se", "significant"), better(r["persist"]["err50"], r["reset"]["err50"])))
    sh = r["shift"]
    out["shift_invariance"] = dict(half_tile=sh["half"], floor=sh["floor"], verdict="PASS" if sh["half"] <= 2 * sh["floor"] + 1e-3 else "FAIL")
    return out


@torch.no_grad()
def shift_error(model, dev, seed, shift):
    b = te.grav_batch("plummer", 200, 16.0, seed, 2, dev)
    a0 = ta.settled_force(model, b)
    d = torch.as_tensor(shift, device=dev, dtype=torch.float64)
    a1 = ta.settled_force(model, dict(b, pos=b["pos"] + d))
    return float(((a1 - a0) ** 2).sum().sqrt() / (a0 ** 2).sum().sqrt())


def run_recurrent(base, dev, seeds):
    arms = dict(reset=dict(persist=False, refresh=False), persist=dict(persist=True, refresh=False), refresh=dict(persist=True, refresh=True))
    res = {k: dict(err50=[], ms=0.0, seam_jump=0.0) for k in arms}
    for k, kw in arms.items():
        tl = ti.Tiled(base, 128.0, sparse=True, **kw)
        for s in seeds:
            if s not in HELD:
                continue
            b = te.grav_batch("multiclump", 300, 128.0, s, 2, dev)
            rt = te.truth_roll(b, base.dt, 50)
            r, secs = clock(lambda: te.model_roll(tl, b, 50), dev.type == "cuda")
            res[k]["err50"].append(float(te.err_curve(r["pos"], rt["pos"], b["mask"])[:, -1].mean()))
            res[k]["ms"] += secs / 50 * 1e3 / max(1, len([x for x in seeds if x in HELD]))
    # seam jump: probe line crossing a tile boundary, force after a few steps of persistent (optionally refreshed) tile state
    res["refresh"]["seam_jump"] = _settled_seam(base, dev, seeds[0], True)
    res["persist"]["seam_jump"] = _settled_seam(base, dev, seeds[0], False)
    tl = ti.Tiled(base, 64.0, sparse=False)
    res["shift"] = dict(half=shift_error(tl, dev, seeds[0], [0.5 * 32.0, 0.0]), floor=shift_error(tl, dev, seeds[0], [1.0, 0.0]))
    res["verdict"] = recurrent_verdict(res)
    return res


@torch.no_grad()
def _settled_seam(base, dev, seed, refresh, steps=3):
    """seam jump / field of the static probe-line force after `steps` forces with persistent (optionally refreshed) tile state."""
    rng = np.random.default_rng(seed)
    tl = ti.Tiled(base, 64.0, sparse=False, persist=True, refresh=refresh)
    xs = np.linspace(-6, 6, 49)
    js = []
    for rep in range(4):
        bp = np.concatenate([rng.uniform([-26, -30], [-8, -12], (100, 2)), rng.uniform([8, -30], [26, -12], (100, 2))])
        pr = np.stack([xs, np.full_like(xs, 0.0)], 1)
        pos = torch.as_tensor(np.concatenate([bp, pr]), device=dev)[None]
        mass = torch.as_tensor(np.concatenate([np.exp(rng.uniform(np.log(0.05), 0.0, 200)), np.full(len(xs), 1e-7)]), device=dev)[None]
        mk = torch.ones_like(mass)
        st = tl.init_state(1, dev)
        for _ in range(steps):
            a, st = tl.force(pos, torch.zeros_like(pos), mass, mk, st, None)
        am = a[0, 200:].cpu().numpy()
        links = np.linalg.norm(np.diff(am, axis=0), axis=1)
        mid = len(xs) // 2 - 1
        js.append(links[mid] / (np.linalg.norm(am, axis=1).mean() + 1e-12))
    return float(np.mean(js))


# ------------------------------------------------------------------ calibration

def spearman(x, y):
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


def calibration_verdict(r):
    """PASS: held-out mean Spearman(pred, actual per-particle force error) >= 0.5, >= 7 of 9 decile steps increasing, and on OOD scenes the predicted error
    rises >= 1.5x whenever the actual error rises >= 1.5x. KILL: Spearman < 0.2 or predicted error does not rise (<1.0x) on OOD scenes where the actual error rose >= 1.5x. Otherwise MIXED."""
    sp = float(np.mean(r["spearman"]))
    mono = float(np.mean(r["decile_increasing"]))
    ood_ok = r["ood_pred_ratio"] >= 1.5 if r["ood_actual_ratio"] >= 1.5 else True
    if sp < 0.2 or (r["ood_actual_ratio"] >= 1.5 and r["ood_pred_ratio"] < 1.0):
        v = "KILL"
    elif sp >= 0.5 and mono >= 7 and ood_ok:
        v = "PASS"
    else:
        v = "MIXED"
    return dict(spearman=sp, deciles_increasing=mono, ood_ok=bool(ood_ok), verdict=v)


@torch.no_grad()
def run_calibration(model, dev, seeds):
    if not hasattr(model, "error"):
        return dict(verdict="skipped: model has no error() method (optional interface); no existing checkpoint trains an error head (TierNet err=False)")
    r = dict(spearman=[], decile_increasing=[])
    for s in seeds:
        b = te.grav_batch("multiclump", 300, ta.SPAN, s, 4, dev)
        st = model.init_state(4, dev)
        ctx = dict(phys=b["phys"])
        pred = model.error(b["pos"], b["vel"], b["mass"], b["mask"], st, ctx)
        a_m = ta.settled_force(model, b)
        act = (a_m - tt.grav_acc(b["pos"], b["mass"], b["mask"])).norm(dim=-1)
        x, y = pred[b["mask"] > 0].cpu().numpy(), act[b["mask"] > 0].cpu().numpy()
        if s in HELD:
            r["spearman"].append(spearman(x, y))
            q = np.quantile(x, np.linspace(0, 1, 11))
            means = [y[(x >= q[i]) & (x <= q[i + 1])].mean() for i in range(10)]
            r["decile_increasing"].append(int(sum(means[i + 1] > means[i] for i in range(9))))
        if s == seeds[0]:
            r["in_pred"], r["in_actual"] = float(x.mean()), float(y.mean())
    bo = te.grav_batch("clumpy", 1200, ta.SPAN, seeds[0], 2, dev)
    bo["mass"] = bo["mass"] * 5.0
    po = model.error(bo["pos"], bo["vel"], bo["mass"], bo["mask"], model.init_state(2, dev), dict(phys=bo["phys"]))
    ao = (ta.settled_force(model, bo) - tt.grav_acc(bo["pos"], bo["mass"], bo["mask"])).norm(dim=-1)
    r["ood_pred_ratio"] = float(po.mean() / r["in_pred"])
    r["ood_actual_ratio"] = float(ao.mean() / r["in_actual"])
    return dict(raw=r, **calibration_verdict(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["respa", "exposure", "sparse", "recurrent", "calibration"])
    ap.add_argument("--model")
    ap.add_argument("--refined")
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[4738, *HELD])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="results/tiered/holes.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if args.mode == "respa":
        out = run_respa(te.load_model(args.model, dev, 0.1), dev, args.seeds, args.steps)
    elif args.mode == "exposure":
        res = {}
        for a in args.arms:
            k, spec = a.split("=", 1)
            m = te.load_model(spec, dev, 0.1)
            res[k] = {str(s): exposure_arm(m, dev, s, args.steps) for s in args.seeds}
            print(k, {s: round(v["err100"], 3) for s, v in res[k].items()}, flush=True)
        out = dict(arms=res, verdict=exposure_verdict(res) if "S" in res else "no reference arm S")
    elif args.mode == "sparse":
        out = run_sparse(te.load_model(args.model, dev, 0.1), dev, args.seeds, te.load_model(args.refined, dev, 0.1) if args.refined else None)
    elif args.mode == "recurrent":
        out = run_recurrent(te.load_model(args.model, dev, 0.1), dev, args.seeds)
    else:
        out = run_calibration(te.load_model(args.model, dev, 0.1), dev, args.seeds)
    print(json.dumps(te.tolist(out.get("verdict", out)), indent=1))
    json.dump(te.tolist(out), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
