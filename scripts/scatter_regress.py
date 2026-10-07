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
TIME_TOL = 0.05
MEM_TOL = 0.10
MEM_DEF = "absolute max_memory_allocated, reset before warm-up"


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def sync():
    torch.cuda.synchronize()


def build_model(args, dev):
    model = tsf.build(tsf.EXPS[args.exp], args.variant, args.dt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
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
    virial = 2K/|U|. c given -> relativistic kinetic energy c^2 (gamma - 1) per unit mass (V is coordinate velocity)."""
    d = P[None, :, :] - P[:, None, :]
    inv = 1.0 / np.sqrt((d ** 2).sum(-1) + eps ** 2)
    np.fill_diagonal(inv, 0.0)
    phi = -(inv * m[None, :]).sum(1)
    v2 = (V ** 2).sum(-1)
    if c is None:
        ke = 0.5 * v2
    else:
        ke = c ** 2 * (1.0 / np.sqrt(1 - np.clip(v2, 0.0, 0.9801 * c ** 2) / c ** 2) - 1)
    K = float((m * ke).sum())
    U = float(0.5 * (m * phi).sum())
    return {"bound_frac": float((ke + phi < 0).mean()), "virial": 2 * K / max(abs(U), 1e-12)}


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
        for _ in range(STEPS):
            mom = mom + 0.5 * dt * a
            pos = pos + dt * speed(mom, c)
            a, field = model.force(pos, speed(mom, c), mass, mask, field)
            mom = mom + 0.5 * dt * a
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
    for k, v in res["timing"].items():
        print(f"timing {k}: {v['tick_ms']:.3f} ms/tick, peak alloc {v['peak_mem_mb']:.1f} MB (reserved {v['peak_reserved_mb']:.1f}, info) (batch {v['batch']}, N {v['n_max']})", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


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
    for k in TIMING_KEYS:
        check(f"{k} tick_ms", base["timing"][k]["tick_ms"], cand["timing"][k]["tick_ms"], TIME_TOL)
        if mem_comparable:
            check(f"{k} peak_mem_mb", base["timing"][k]["peak_mem_mb"], cand["timing"][k]["peak_mem_mb"], MEM_TOL)
    print(f"baseline {base['meta']['ckpt']} @ {base['meta']['git']}  candidate {cand['meta']['ckpt']} @ {cand['meta']['git']}")
    print(f"{'metric':32s} {'baseline':>12s} {'candidate':>12s} {'delta%':>8s}")
    for name, b, c, d, v in rows:
        print(f"{name:32s} {b:12.5g} {c:12.5g} {d:8.1f}  {v}")
    info = [f"{reg} err@{k} {base['regimes'][reg]['mean']['err'][k]:.4g} -> {cand['regimes'][reg]['mean']['err'][k]:.4g}" for reg in REGIMES for k in ("50", "100")]
    print("info (not gated):", "; ".join(info))
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
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
