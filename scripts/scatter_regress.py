"""Frozen regression harness for the scatter-field model.

Run: for a checkpoint, held-out seeds (9100, 9200, 9300, 4738), three regimes (two_body: 2 unit-mass bodies, 48 scenes/seed; expB: 10-100 bodies scale-init,
48 scenes/seed; bh300: 300-body relativistic BH collapse c=10 as in scatter_bh.py, 1 scene/seed), 100-tick rollouts. Reports err@5/10/20/50/100 (mean per-body
position error vs truth, sim units), dE/E, |dP|, dL/L at tick 100 (model and truth), CUDA-synchronised warmed-up per-tick wall time and peak GPU memory
(over the rollout's starting allocation). Compare: PASS/FAIL of a candidate JSON against a baseline JSON.

Usage:
  PYTHONPATH=. python scripts/scatter_regress.py run --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/scatter_baseline.json
  PYTHONPATH=. python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate results/cand.json
"""
import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import scatter_ood as so
from scripts import train_scatter_field as tsf

SEEDS = (9100, 9200, 9300, 4738)
TICKS = (5, 10, 20, 50, 100)
STEPS = 100
SCENES = 48
TIMING_REPS = 5
BH = dict(n=300, sigma=8.0, vfac=0.3, c=10.0)
REGIMES = ("two_body", "expB", "bh300")
TIMING_KEYS = REGIMES + ("expB_batch1_n100",)
ACC_TOL = 0.15
ACC_TOL_KEY = {("two_body", "5"): 0.30}
ACC_ATOL = 1e-5
STATE_ATOL = {"bound_frac": 0.02, "virial": 0.05}
CONS_STEPS = 1000
CONS_REGIMES = ("nr100", "bh300")
CONS_TOL = 0.25
TIME_TOL = 0.05
MEM_TOL = 0.10
INFO_TICKS = {"two_body": ("50", "100"), "expB": ("50", "100"), "bh300": ("50",)}
FLOOR_TICKS = INFO_TICKS
MEM_DEF = "absolute max_memory_allocated, reset before warm-up"


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def sync():
    torch.cuda.synchronize()


class CFHarness(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.extent = inner, inner.extent

    def init_field(self, B, device, dtype=torch.float32):
        return None, None

    def step(self, pos, vel, mass, mask, field):
        return self.inner.step(pos, vel, mass, mask, field)

    def force(self, pos, vel, mass, mask, field):
        return self.inner.accel(pos, mass, mask), field


def build_model(args, dev):
    if args.variant == "cf":
        from scripts.gap_probe import CFDense
        inner = CFDense().to(dev)
        inner.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
        inner.eval()
        return CFHarness(inner)
    model = tsf.build(tsf.EXPS[args.exp], args.variant, args.dt).to(dev)
    sf.load_compat(model, torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    for o in [x for x in args.opt.split(",") if x]:
        sf.apply_opt(model, o)
    return model


def timed(fn, steps, dev):
    """Reset peak stats, warm up once (includes CUDA-graph capture), then median per-tick ms over TIMING_REPS runs.
    Memory (MB) = absolute max_memory_allocated over warm-up + reps (same definition for eager and graphed); reserved is info only."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    fn()
    sync()
    ts = []
    for _ in range(TIMING_REPS):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) / steps * 1000)
    return {"tick_ms": float(np.median(ts)), "tick_ms_all": ts, "peak_mem_mb": torch.cuda.max_memory_allocated(dev) / 2 ** 20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(dev) / 2 ** 20}


def tick_values(xs, offset):
    return {str(k): float(xs[k - offset]) for k in TICKS}


def state_summary(P, V, m, eps, c=None):
    """One snapshot, numpy P, V (N,2), masses m (N,). Bound fraction = share of bodies with KE_i + phi_i < 0 (phi_i softened potential from all others),
    virial = 2K/|U|. c given -> relativistic kinetic energy c^2 (gamma - 1) per unit mass and momenta gamma v (V is coordinate velocity).
    Also returns E = K + U, total momentum, angular momentum about the mass centre (used by chaos_floors.py)."""
    d = P[None, :, :] - P[:, None, :]
    inv = 1.0 / np.sqrt((d ** 2).sum(-1) + eps ** 2)
    np.fill_diagonal(inv, 0.0)
    phi = -(inv * m[None, :]).sum(1)
    v2 = (V ** 2).sum(-1)
    if c is None:
        ke = 0.5 * v2
        mom = m[:, None] * V
    else:
        gamma = 1.0 / np.sqrt(1 - np.clip(v2, 0.0, 0.9801 * c ** 2) / c ** 2)
        ke = c ** 2 * (gamma - 1)
        mom = m[:, None] * gamma[:, None] * V
    K = float((m * ke).sum())
    U = float(0.5 * (m * phi).sum())
    r = P - (m[:, None] * P).sum(0) / m.sum()
    return {"bound_frac": float((ke + phi < 0).mean()), "virial": 2 * K / max(abs(U), 1e-12), "E": K + U, "P": mom.sum(0),
            "L": float((r[:, 0] * mom[:, 1] - r[:, 1] * mom[:, 0]).sum())}


def tag_state(out, tag, P, V, m, eps, c=None):
    summ = [state_summary(P[i], V[i], m[i], eps, c) for i in range(len(P))]
    for key in ("bound_frac", "virial"):
        out[f"{key}_{tag}"] = float(np.mean([x[key] for x in summ]))


def metrics_ab(model, data, dev, eps):
    P, V, M, mask = sf.to_tensors(data, dev)
    Pm, Vm = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), STEPS)
    m = sf.traj_metrics(Pm.double(), Vm.double(), P, V, M, mask, eps)
    out = {"err": tick_values(m["err"], 1)}
    scene_err = (((Pm.double() - P) ** 2).sum(-1).sqrt() * mask[:, None]).sum(2) / mask.sum(1)[:, None]
    out["err_sem"] = {str(k): float(scene_err[:, k].std() / np.sqrt(scene_err.shape[0])) for k in TICKS}
    for tag in ("model", "true"):
        out[f"dE_rel_{tag}"] = float(m[f"energy_drift_rel_{tag}"][STEPS])
        out[f"dP_{tag}"] = float(m[f"mom_drift_{tag}"][STEPS])
        out[f"dL_rel_{tag}"] = float(m[f"angmom_drift_rel_{tag}"][STEPS])
    Mn, kn = M.cpu().numpy(), mask.cpu().numpy().astype(bool)
    for tag, PP, VV in (("model", Pm.double(), Vm.double()), ("true", P, V)):
        Pn, Vn = PP[:, STEPS].cpu().numpy(), VV[:, STEPS].cpu().numpy()
        tag_state(out, tag, [Pn[i][kn[i]] for i in range(len(Pn))], [Vn[i][kn[i]] for i in range(len(Pn))], [Mn[i][kn[i]] for i in range(len(Pn))], eps)
    return out, (P, M, mask)


def conservation_rows(model, dev, args):
    """Standing long-horizon rows (CONS_STEPS ticks): median over scenes of |dE|/(K0+|U0|), |dL|/Lscale, |dP|, bound fraction, virial, outside-grid fraction,
    max |v|/c and fraction with |v|>=c (bh only), model and exact-sim truth (4 substeps), regimes from scatter_conservation."""
    from scripts import scatter_conservation as sc

    out = {}
    for name in CONS_REGIMES:
        kind, n, per = sc.REGIMES[name]
        pos0, vel0, seeds = sc.make_scenes(kind, n, per)
        P0 = torch.tensor(pos0, device=dev)
        V0 = torch.tensor(vel0, device=dev)
        rel = kind == "bh"
        r0 = (P0 - P0.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()
        recs = {}
        for tag in ("model", "true"):
            rec = sc.Recorder(rel, args.eps, model.extent, r0, CONS_STEPS, len(seeds), dev, args.dt)
            if tag == "model":
                sc.roll_model(model, P0, V0, rel, CONS_STEPS, rec, None, args.dt)
            else:
                sc.roll_truth(P0, V0, rel, CONS_STEPS, 4, args.eps, rec, None, args.dt)
            recs[tag] = rec.rows
        res = {}
        for tag, r in recs.items():
            K0, U0 = r["K"][0], r["U"][0]
            lsc = r0 * torch.sqrt(2 * n * K0.clamp_min(1e-12))
            t = CONS_STEPS
            res[tag] = {"dE": float(((r["E"][t] - r["E"][0]).abs() / (K0 + U0.abs())).median()),
                        "dL": float(((r["L"][t] - r["L"][0]).abs() / lsc).median()), "dP": float((r["P"][t] - r["P"][0]).abs().median()),
                        "bound": float(r["bound"][t].median()), "vir": float(r["vir"][t].median()), "outgrid": float(r["outgrid"][t].median()),
                        "vmax_over_c": float(r["vmax"].amax() / sc.C) if rel else None, "frac_ge_c": float(r["fracge"].amax()) if rel else None,
                        "finite": float(r["finite"].min())}
        res["bound_gap"] = abs(res["model"]["bound"] - res["true"]["bound"])
        res["vir_gap"] = abs(float(np.log(max(res["model"]["vir"], 1e-9) / max(res["true"]["vir"], 1e-9))))
        out[name] = res
        print(f"cons {name} {CONS_STEPS}t dE {res['model']['dE']:.4g} (true {res['true']['dE']:.4g}) dL {res['model']['dL']:.4g} bound {res['model']['bound']:.3f}/{res['true']['bound']:.3f} "
              f"vir {res['model']['vir']:.3g}/{res['true']['vir']:.3g} outgrid {res['model']['outgrid']:.3f}", flush=True)
    return out


def bh_scene(seed):
    rng = np.random.default_rng(seed)
    n = BH["n"]
    pos0 = np.clip(rng.normal(0.0, BH["sigma"], (n, 2)), -29.0, 29.0)
    vel0 = rng.normal(0.0, BH["vfac"] * np.sqrt(n / (4 * BH["sigma"])), (n, 2))
    vel0 -= vel0.mean(0)
    return pos0, vel0


def bh_rollout(model, pos0, vel0, dev, dt):
    c = BH["c"]
    n = pos0.shape[0]
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
    field = model.init_field(1, dev)[0]
    ps, vs = [pos[0]], [speed(mom, c)[0]]
    with torch.no_grad():
        a, field = model.force(pos, speed(mom, c), mass, mask, field)
        sub = getattr(model, "substeps", 1)
        h = dt / sub
        for _ in range(STEPS):
            for _ in range(sub):
                mom = mom + 0.5 * h * a
                pos = pos + h * speed(mom, c)
                a, field = model.force(pos, speed(mom, c), mass, mask, field)
                mom = mom + 0.5 * h * a
            ps.append(pos[0])
            vs.append(speed(mom, c)[0])
    return torch.stack(ps), torch.stack(vs)


def rel_conserved(P, V, eps, c):
    """P, V (T,N,2) numpy, unit masses, relativistic. Returns dE/E, |dP|, dL/L at the last tick (E per gs.energy_rel; P, L from momenta p = gamma v)."""
    vm2 = np.clip((V ** 2).sum(-1, keepdims=True), 0.0, 0.9801 * c ** 2)
    mom = V / np.sqrt(1 - vm2 / c ** 2)
    E = np.array([gs.energy_rel(P[t], V[t], eps, c) for t in (0, -1)])
    Ptot = mom.sum(1)
    com = P.mean(1, keepdims=True)
    r = P - com
    L = (r[..., 0] * mom[..., 1] - r[..., 1] * mom[..., 0]).sum(1)
    return (float(abs(E[1] - E[0]) / max(abs(E[0]), 1e-9)), float(np.linalg.norm(Ptot[-1] - Ptot[0])),
            float(abs(L[-1] - L[0]) / max(abs(L[0]), 1e-3)))


def metrics_bh(model, seed, dev, args):
    pos0, vel0 = bh_scene(seed)
    ps, vs = bh_rollout(model, pos0, vel0, dev, args.dt)
    Pm, Vm = ps.double().cpu().numpy(), vs.double().cpu().numpy()
    Pt, Vt = gs.rollout_torch(pos0, vel0, STEPS, dev, dt=args.dt, substeps=4, eps=args.eps, relativistic=True, c=BH["c"])
    Pt, Vt = np.asarray(Pt), np.asarray(Vt)
    err = np.linalg.norm(Pm - Pt, axis=-1).mean(1)
    out = {"err": tick_values(err, 0), "vmax_over_c": float(np.linalg.norm(Vm, axis=-1).max() / BH["c"]),
           "vmax_over_c_true": float(np.linalg.norm(Vt, axis=-1).max() / BH["c"]),
           "frac_ge_0p9c": float((np.linalg.norm(Vm, axis=-1) >= 0.9 * BH["c"]).mean()),
           "frac_ge_0p9c_true": float((np.linalg.norm(Vt, axis=-1) >= 0.9 * BH["c"]).mean())}
    ones = np.ones((1, Pm.shape[1]))
    tag_state(out, "model", Pm[-1:], Vm[-1:], ones, args.eps, BH["c"])
    tag_state(out, "true", Pt[-1:], Vt[-1:], ones, args.eps, BH["c"])
    for tag, P, V in (("model", Pm, Vm), ("true", Pt, Vt)):
        out[f"dE_rel_{tag}"], out[f"dP_{tag}"], out[f"dL_rel_{tag}"] = rel_conserved(P, V, args.eps, BH["c"])
    return out


def aggregate(per_seed):
    keys = list(per_seed.values())[0].keys()
    mean = {}
    for k in keys:
        if isinstance(per_seed[SEEDS[0]][k], dict):
            mean[k] = {t: float(np.mean([per_seed[s][k][t] for s in per_seed])) for t in per_seed[SEEDS[0]][k]}
        else:
            mean[k] = float(np.mean([per_seed[s][k] for s in per_seed]))
    if "err_sem" in per_seed[SEEDS[0]]:
        mean["err_sem"] = {t: float(np.sqrt(sum(per_seed[s]["err_sem"][t] ** 2 for s in per_seed)) / len(per_seed)) for t in per_seed[SEEDS[0]]["err"]}
    else:
        mean["err_sem"] = {t: float(np.std([per_seed[s]["err"][t] for s in per_seed]) / np.sqrt(len(per_seed))) for t in per_seed[SEEDS[0]]["err"]}
    spread = {t: float(np.std([per_seed[s]["err"][t] for s in per_seed]) / np.mean([per_seed[s]["err"][t] for s in per_seed])) for t in per_seed[SEEDS[0]]["err"]}
    return mean, spread


BATTERY = [("ref", "cluster_N100_s8_v0.3"), ("mass_ratio_2body", "binary_q10"), ("mass_dominant", "cluster_heavy30"),
           ("mass_spectrum", "logunif_0.1_10"), ("mass_scale_equal", "equal_m3.0"), ("mass_scale_equal", "equal_m10.0"),
           ("density_sigma", "sigma2.0"), ("density_plummer", "plummer_a0.25"), ("softening_eps", "cluster_eps0.1"),
           ("softening_eps", "cluster_eps2.0"), ("boundary_offset", "cluster_shift24.0"), ("dt", "cluster_dt0.4"),
           ("velocity_scale", "vfac4.0"), ("ic_shape", "filament"), ("binary_tight", "binary_d0.25")]
BATTERY_SEEDS = (4738, 9100)


def battery(model, dev, args):
    """Compact OOD battery (decision #6.5): err@2 time units per point (mean of BATTERY_SEEDS), non-gating."""
    specs = so.specs()
    out = {}
    for ax, name in BATTERY:
        spec = [x for x in specs[ax] if x["name"] == name][0]
        spec["axis"] = ax
        rows = [so.run_point(model, spec, sd, dev, args, None) for sd in BATTERY_SEEDS]
        out[f"{ax}/{name}"] = {"err@2": float(np.mean([r["err@2.0"] for r in rows])), "floor@2": float(np.mean([r["floor@2.0"] for r in rows])),
                               "ballistic@2": float(np.mean([r["ballistic@2.0"] for r in rows])), "nonfinite": any(r["nonfinite_tick"] is not None for r in rows)}
        print(f"battery {ax}/{name} err@2 {out[f'{ax}/{name}']['err@2']:.4g} floor {out[f'{ax}/{name}']['floor@2']:.2g}", flush=True)
    return out


def run(args):
    dev = torch.device("cuda")
    model = build_model(args, dev)
    res = {"meta": {"ckpt": args.ckpt, "exp": args.exp, "variant": args.variant, "dt": args.dt, "eps": args.eps, "seeds": list(SEEDS),
                    "opt": args.opt, "steps": STEPS, "scenes_per_seed": SCENES, "bh": BH, "timing_reps": TIMING_REPS, "mem_def": MEM_DEF, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(dev),
                    "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
           "regimes": {}, "timing": {}}
    cfgs = {"two_body": dict(ball_range=(2, 2), scale=False), "expB": dict(ball_range=(10, 100), scale=True)}
    timing_data = {}
    for reg in REGIMES:
        per_seed = {}
        for seed in SEEDS:
            if reg == "bh300":
                per_seed[seed] = metrics_bh(model, seed, dev, args)
            else:
                data = gs.make_dataset(SCENES, cfgs[reg]["ball_range"], STEPS, seed, scale=cfgs[reg]["scale"], device=dev, dt=args.dt, eps=args.eps)
                per_seed[seed], _ = metrics_ab(model, data, dev, args.eps)
                if seed == SEEDS[-1]:
                    timing_data[reg] = data
            print(f"{reg} seed {seed} err@{list(TICKS)} {[round(per_seed[seed]['err'][str(k)], 5) for k in TICKS]}", flush=True)
        mean, spread = aggregate(per_seed)
        res["regimes"][reg] = {"per_seed": {str(s): v for s, v in per_seed.items()}, "mean": mean, "err_rel_std_across_seeds": spread}
        print(f"{reg} MEAN err {[round(mean['err'][str(k)], 5) for k in TICKS]} dE/E {mean['dE_rel_model']:.4g} (true {mean['dE_rel_true']:.4g})", flush=True)
    for reg in ("two_body", "expB"):
        P, V, M, mask = sf.to_tensors(timing_data[reg], dev)
        args_in = (P[:, 0].float(), V[:, 0].float(), M.float(), mask.float())
        res["timing"][reg] = timed(lambda: sf.rollout(model, *args_in, STEPS), STEPS, dev)
        res["timing"][reg].update(batch=P.shape[0], n_max=P.shape[2])
    pos0, vel0 = bh_scene(SEEDS[-1])
    res["timing"]["bh300"] = timed(lambda: bh_rollout(model, pos0, vel0, dev, args.dt), STEPS, dev)
    res["timing"]["bh300"].update(batch=1, n_max=BH["n"])
    rng = np.random.default_rng(SEEDS[-1])
    p1, v1 = gs.init_bodies(100, rng, scale=True)
    P1, V1, M1, k1 = sf.to_tensors([(p1[None], v1[None])], dev)
    a1 = (P1[:, 0].float(), V1[:, 0].float(), M1.float(), k1.float())
    res["timing"]["expB_batch1_n100"] = timed(lambda: sf.rollout(model, *a1, STEPS), STEPS, dev)
    res["timing"]["expB_batch1_n100"].update(batch=1, n_max=100)
    if not args.no_cons:
        res["conservation"] = conservation_rows(model, dev, args)
    for k, v in res["timing"].items():
        print(f"timing {k}: {v['tick_ms']:.3f} ms/tick, peak alloc {v['peak_mem_mb']:.1f} MB (reserved {v['peak_reserved_mb']:.1f}, info) (batch {v['batch']}, N {v['n_max']})", flush=True)
    if args.battery:
        res["ood_battery"] = battery(model, dev, args)
        ref = res["ood_battery"]["ref/cluster_N100_s8_v0.3"]["err@2"]
        res["ood_battery_summary"] = {"geomean_ratio_to_ref": float(np.exp(np.mean([np.log(max(v["err@2"], 1e-9) / ref) for v in res["ood_battery"].values()])))}
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


def print_floors(cand, floors_path):
    """Candidate metrics against the reference-noise floor (sub_x2: twice the substeps) and chaos floor (1e-5 relative position perturbation) of
    results/chaos_floors.json (scripts/chaos_floors.py). Not gated. 'WITHIN' = candidate <= reference-noise floor, i.e. indistinguishable from the reference."""
    try:
        floors = json.load(open(floors_path))["groups"]
    except FileNotFoundError:
        print(f"floors: {floors_path} not found, skipped")
        return
    print(f"{'floors (not gated)':28s} {'candidate':>10s} {'ref-noise':>10s} {'chaos1e-5':>10s} {'c/ref':>8s} {'c/chaos':>8s}")
    for reg in REGIMES:
        if reg not in floors:
            continue
        v, m = floors[reg]["variants"], cand["regimes"][reg]["mean"]
        rows = [(f"{reg} err@{k}", m["err"][k], v["sub_x2"]["err"][k], v["pert_1e-05"]["err"][k]) for k in ("5", "10", "20") + FLOOR_TICKS[reg]]
        rows += [(f"{reg} dE/E@100", m["dE_rel_model"], max(v["ref"]["dE_rel"]["100"], v["sub_x2"]["dE_rel"]["100"]), v["pert_1e-05"]["dE_rel"]["100"]),
                 (f"{reg} dP@100", m["dP_model"], max(v["ref"]["dP"]["100"], v["sub_x2"]["dP"]["100"]), v["pert_1e-05"]["dP"]["100"]),
                 (f"{reg} dL/L@100", m["dL_rel_model"], max(v["ref"]["dL_rel"]["100"], v["sub_x2"]["dL_rel"]["100"]), v["pert_1e-05"]["dL_rel"]["100"])]
        if "bound_frac_true" in m:
            rows += [(f"{reg} bound gap@100", abs(m["bound_frac_model"] - m["bound_frac_true"]), v["sub_x2"]["bound_gap"]["100"], v["pert_1e-05"]["bound_gap"]["100"]),
                     (f"{reg} virial gap@100", abs(m["virial_model"] / m["virial_true"] - 1), v["sub_x2"]["virial_gap"]["100"], v["pert_1e-05"]["virial_gap"]["100"])]
        for name, c, f_ref, f_chaos in rows:
            ratio = lambda f: f"{c / f:8.3g}" if f > 1e-12 else f"{'n/a':>8s}"
            print(f"{name:28s} {c:10.4g} {f_ref:10.3g} {f_chaos:10.3g} {ratio(f_ref)} {ratio(f_chaos)}  {'WITHIN' if c <= f_ref else ''}")


def compare(args):
    base, cand = json.load(open(args.baseline)), json.load(open(args.candidate))
    rows, ok = [], True

    def check(name, b, c, tol, atol=0.0):
        nonlocal ok
        good = bool(c <= b * (1 + tol) + atol)
        ok &= good
        rows.append((name, b, c, (c / b - 1) * 100 if b else float("nan"), "PASS" if good else "FAIL"))

    mem_comparable = base["meta"].get("mem_def") == cand["meta"].get("mem_def") == MEM_DEF
    if not mem_comparable:
        print(f"memory gate SKIPPED: definitions differ (baseline {base['meta'].get('mem_def')!r}, candidate {cand['meta'].get('mem_def')!r})")
    for reg in REGIMES:
        for k in ("5", "10", "20"):
            check(f"{reg} err@{k}", base["regimes"][reg]["mean"]["err"][k], cand["regimes"][reg]["mean"]["err"][k], ACC_TOL_KEY.get((reg, k), ACC_TOL), ACC_ATOL)
    for reg in REGIMES:
        bm, cm = base["regimes"][reg]["mean"], cand["regimes"][reg]["mean"]
        if "bound_frac_true" not in bm:
            print(f"{reg}: baseline lacks bound/virial rows, skipped")
            continue
        for name, f in (("bound_frac", lambda m: abs(m["bound_frac_model"] - m["bound_frac_true"])),
                        ("virial", lambda m: abs(m["virial_model"] / m["virial_true"] - 1))):
            check(f"{reg} {name} gap@100", f(bm), f(cm), ACC_TOL, STATE_ATOL[name])
    if "vmax_over_c" in cand["regimes"]["bh300"]["mean"]:
        v = cand["regimes"]["bh300"]["mean"]["vmax_over_c"]
        rows.append(("bh300 vmax/c (<1)", base["regimes"]["bh300"]["mean"].get("vmax_over_c", float("nan")), v, float("nan"), "PASS" if v < 1.0 else "FAIL"))
        ok &= v < 1.0
        for k, atol in (("dE_rel_model", 1e-4), ("dL_rel_model", 1e-3), ("dP_model", 1e-3)):
            if k in base["regimes"][reg]["mean"] and k in cand["regimes"][reg]["mean"]:
                check(f"{reg} {k}@100", base["regimes"][reg]["mean"][k], cand["regimes"][reg]["mean"][k], CONS_TOL, atol)
    for reg, bc in base.get("conservation", {}).items():
        cc = cand.get("conservation", {}).get(reg)
        if cc is None:
            continue
        for k, b, c, atol in (("dE", bc["model"]["dE"], cc["model"]["dE"], 1e-3), ("dL", bc["model"]["dL"], cc["model"]["dL"], 1e-2),
                              ("dP", bc["model"]["dP"], cc["model"]["dP"], 1e-3), ("bound_gap", bc["bound_gap"], cc["bound_gap"], 0.02),
                              ("vir_gap", bc["vir_gap"], cc["vir_gap"], 0.05), ("outgrid", bc["model"]["outgrid"], cc["model"]["outgrid"], 0.01)):
            check(f"cons {reg}@{CONS_STEPS} {k}", b, c, CONS_TOL, atol)
        if cc["model"]["frac_ge_c"] is not None:
            check(f"cons {reg} frac|v|>=c", 0.0, cc["model"]["frac_ge_c"], 0.0)
        check(f"cons {reg} nonfinite", 0.0, 1.0 - cc["model"]["finite"], 0.0)
    for k in TIMING_KEYS:
        check(f"{k} tick_ms", base["timing"][k]["tick_ms"], cand["timing"][k]["tick_ms"], TIME_TOL)
        if mem_comparable:
            check(f"{k} peak_mem_mb", base["timing"][k]["peak_mem_mb"], cand["timing"][k]["peak_mem_mb"], MEM_TOL)
    print(f"baseline {base['meta']['ckpt']} @ {base['meta']['git']}  candidate {cand['meta']['ckpt']} @ {cand['meta']['git']}")
    print(f"{'metric':32s} {'baseline':>12s} {'candidate':>12s} {'delta%':>8s}")
    for name, b, c, d, v in rows:
        print(f"{name:32s} {b:12.5g} {c:12.5g} {d:8.1f}  {v}")
    if "ood_battery" in cand:
        print("OOD battery (non-gating), err@2 time units: baseline -> candidate")
        for k, v in cand["ood_battery"].items():
            b = base.get("ood_battery", {}).get(k)
            print(f"  {k:42s} {'%.4g' % b['err@2'] if b else '-':>10s} -> {v['err@2']:.4g}")
        print("  geomean ratio to ref:", base.get("ood_battery_summary", {}).get("geomean_ratio_to_ref"), "->", cand["ood_battery_summary"]["geomean_ratio_to_ref"])
    info = [f"{reg} err@{k} {base['regimes'][reg]['mean']['err'][k]:.4g} -> {cand['regimes'][reg]['mean']['err'][k]:.4g}" for reg in REGIMES for k in INFO_TICKS[reg]]
    print("info (not gated; bh300 err@100 is at the chaos ceiling and not quoted):", "; ".join(info))
    print_floors(cand, args.floors)
    print("OVERALL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--exp", default="E")
    r.add_argument("--variant", default="ms_kp_pot_v_g128")
    r.add_argument("--dt", type=float, default=0.1)
    r.add_argument("--eps", type=float, default=0.5)
    r.add_argument("--opt", default="")
    r.add_argument("--no-cons", action="store_true")
    r.add_argument("--battery", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--floors", default="results/chaos_floors.json")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
