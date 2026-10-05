"""Train/evaluate the scatter-field dynamics model on 2D N-body gravity.
Exp A: 2 unit-mass bodies. Exp B: 10-100 unit-mass bodies (scale-init). Exp C: star + light bodies (mixed masses).
Data gen, truth sim and the err@k eval reuse scripts/gravity_sim.py so numbers are comparable with
scripts/train_gravity_dynamics.py (train seed 4738, held-out eval seeds 9000/12000, 48 scenes)."""
import argparse
import json
import os
import time

import numpy as np
import torch

from model.central_force import CentralForceDynamics
from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts.train_gravity_dynamics import unroll

BASE = dict(grid=64, net="local", recurrent=True, hidden_ch=0, momfix=True)
VARIANTS = {
    "ss_rec": dict(),
    "ss_norec": dict(recurrent=False),
    "ms_rec": dict(net="unet"),
    "ms_norec": dict(net="unet", recurrent=False),
    "ms_rec_g32": dict(net="unet", grid=32),
    "ms_rec_g128": dict(net="unet", grid=128),
    "ms_rec_nomomfix": dict(net="unet", momfix=False),
    "ss_rec_h8": dict(hidden_ch=8),
    "ss_pot": dict(potential=True),
    "ms_ker": dict(net="unet", kernel=True),
    "ms_ker_g128": dict(net="unet", kernel=True, grid=128),
    "ms_ker_pp": dict(net="unet", kernel=True, pp=2.0),
    "ms_ker_pp_g128": dict(net="unet", kernel=True, pp=2.0, grid=128),
    "kp_nonet": dict(net="unet", kernel=True, pp=2.0, split=True, nonet=True),
    "ms_kp_split": dict(net="unet", kernel=True, pp=2.0, split=True),
    "ms_kp_pot": dict(net="unet", kernel=True, pp=2.0, split=True, potential=True),
    "ms_kp_pot_g128": dict(net="unet", kernel=True, pp=2.0, split=True, potential=True, grid=128),
    "ms_kp_pot_g128": dict(net="unet", kernel=True, pp=2.0, split=True, potential=True, grid=128),
    "ms_kp_pot_v": dict(net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True),
    "ms_kp_pot_v_g128": dict(net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, grid=128),
    "ms_pot": dict(net="unet", potential=True),
    "ms_pot_g128": dict(net="unet", potential=True, grid=128),
    "ml_rec": dict(arch="ml"),
    "ml_norec": dict(arch="ml", recurrent=False),
    "ml_noquad": dict(arch="ml", quad=False),
    "ml_k4": dict(arch="ml", k_leaf=4),
    "ml_nodm": dict(arch="ml", dm_apply=False),
}
ML_BASE = dict(quad=True, recurrent=True, hidden_ch=0, momfix=True, k_leaf=1, dm_apply=True)
EXPS = {
    "A": dict(n=(2, 2), scale=False, extent=32.0, in_scale=1.0, star=False,
              variants=["ss_rec", "ss_norec", "ms_rec", "ms_norec", "ms_rec_g32", "ms_rec_g128", "ms_rec_nomomfix", "ss_rec_h8", "ml_rec", "ml_norec", "ml_noquad", "ml_k4", "ml_nodm", "ss_pot", "ms_pot", "ms_pot_g128", "ms_ker", "ms_ker_g128", "ms_ker_pp", "ms_ker_pp_g128", "kp_nonet", "ms_kp_split", "ms_kp_pot", "ms_kp_pot_g128", "ms_kp_pot_v", "ms_kp_pot_v_g128"]),
    "B": dict(n=(10, 100), scale=True, extent=64.0, in_scale=1.0, star=False,
              variants=["ss_rec", "ss_norec", "ms_rec", "ms_norec", "ms_rec_g32", "ms_rec_g128", "ms_rec_nomomfix", "ml_rec", "ml_norec", "ml_k4"]),
    "E": dict(n=(10, 200), scale=True, extent=64.0, in_scale=1.0, star=False, cluster=0.5, variants=["ms_kp_pot_v_g128"]),
    "C": dict(n=None, scale=False, extent=64.0, in_scale=0.1, star=True,
              variants=["ss_rec", "ss_norec", "ms_rec", "ms_norec", "ss_rec_h8", "ml_rec", "ml_norec"]),
    "D": dict(n=(2, 2), scale=False, extent=32.0, in_scale=1.0, star=False, orbit=True,
              variants=["ss_rec", "ms_rec", "ml_rec", "ml_noquad", "ml_k4", "ml_nodm"]),
}
EVAL_SEEDS = (9000, 12000)


def make_data(cfg, num, steps, seed, device, dt, eps):
    if cfg["star"]:
        return sf.make_star_dataset(num, steps, seed, device, dt=dt, eps=eps)
    if cfg.get("orbit"):
        return sf.make_orbit_dataset(num, steps, seed, device, dt=dt, eps=eps)
    if cfg.get("cluster"):
        k = int(num * cfg["cluster"])
        return (gs.make_dataset(num - k, cfg["n"], steps, seed, scale=cfg["scale"], device=device, dt=dt, eps=eps)
                + sf.make_cluster_dataset(k, steps, seed + 1, device, dt=dt, eps=eps))
    return gs.make_dataset(num, cfg["n"], steps, seed, scale=cfg["scale"], device=device, dt=dt, eps=eps)


def build(cfg, name, dt):
    if VARIANTS[name].get("arch") == "ml":
        kw = {**ML_BASE, **{k: v for k, v in VARIANTS[name].items() if k != "arch"}}
        return sf.MultiLevelScatterField(extent=cfg["extent"], h_leaf=0.5, dt=dt, in_scale=cfg["in_scale"], **kw)
    kw = dict(BASE)
    kw.update(VARIANTS[name])
    return sf.ScatterField(extent=cfg["extent"], dt=dt, in_scale=cfg["in_scale"], **kw)


def train(model, tensors, args, ckpt_path, log):
    P, V, M, mask = [t.float() for t in tensors]
    S = P.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = 0
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["it"]
        log(f"resumed from it {start}")
    g = torch.Generator(device="cpu").manual_seed(args.seed + start)
    fns = sf.make_step_fns(model, args.batch, P.shape[2], args.k_end, P.device) if "graphtrain" in args.opt.split(",") else None
    skipped = 0
    t_start = time.time()
    it = start - 1
    while True:
        it += 1
        if args.time_budget > 0:
            frac = (time.time() - t_start) / args.time_budget
            if frac >= 1.0:
                break
        else:
            if it >= args.iters:
                break
            frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(S, (args.batch,), generator=g).to(args.device)
        t0 = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(args.device)
        tt = t0[:, None] + torch.arange(k + 1, device=args.device)[None]
        Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
        m, mk = M[idx], mask[idx]
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = model.init_field(args.batch, args.device)
        lp = lv = lm = 0.0
        w = mk[:, :, None]
        denom = mk.sum() * 2 * k
        for s in range(k):
            pos, vel, m, field, _ = fns[s](pos, vel, m, mk, field) if fns else sf._step_m(model, pos, vel, m, mk, field)
            lm = lm + (((m - M[idx]) ** 2) * mk).sum() / (mk.sum() * k)
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w).sum() / denom
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
        loss = lp + 0.1 * lv + lm
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        cap = getattr(args, 'loss_cap', 0.0)
        if torch.isfinite(loss) and torch.isfinite(gn) and (cap <= 0 or loss.item() < cap):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} non-finite loss/grad, step skipped ({skipped} total)")
        if it % args.log_every == 0:
            log(f"it {it} k {k} loss {loss.item():.6f} t {time.time() - t_start:.0f}s")
            if all(torch.isfinite(p).all() for p in model.parameters()):
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it + 1}, ckpt_path)
                if it % 5000 == 0:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it + 1}, f"{ckpt_path}.it{it}")
    if all(torch.isfinite(p).all() for p in model.parameters()):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it}, ckpt_path)
    log(f"trained {it} iters in {time.time() - t_start:.0f}s")
    return it, time.time() - t_start


def headline(m):
    return [round(m["err"][i], 4) for i in (4, 9, 19)]


def evaluate_model(model, data, device, dt, eps):
    P, V, M, mask = sf.to_tensors(data, device)
    Pm, Vm = sf.rollout(model, P[:, 0].float(), V[:, 0].float(), M.float(), mask.float(), P.shape[1] - 1)
    Pm, Vm = Pm.double(), Vm.double()
    out = sf.traj_metrics(Pm, Vm, P, V, M, mask, eps)
    out["const_vel_err"] = sf.const_vel_err(P, V, mask, dt)
    mk = mask[..., None]
    cnt = mask.sum() 
    chan = {}
    for tag, A, Bt in (("pos", Pm, P), ("vel", Vm, V)):
        for ax, nm in ((0, "x"), (1, "y")):
            e2 = ((A[..., ax] - Bt[..., ax]) ** 2 * mask[:, None]).sum((0, 2)) / cnt
            chan[f"{tag}_{nm}"] = [float(e2[k].sqrt()) for k in (5, 10, 20)]
    out["chan_rms_at_5_10_20"] = chan
    out["mass_drift"] = sf.rollout.last_dm.cpu().tolist()
    box = model.extent / 2
    out["frac_in_grid_final"] = float((((Pm[:, -1].abs() < box).all(-1).double() * mask).sum(1) / mask.sum(1)).mean())
    return out, (Pm, Vm, P, V)


SWEEP_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def sweep(model, dev, args, baseline=False):
    out = {}
    for r in SWEEP_RATIOS:
        data = sf.make_orbit_dataset(16, 100, 4738, dev if dev.type == "cuda" else None, fixed_ratio=r, dt=args.dt, eps=args.eps)
        if baseline:
            m, (Pm, _, Pt, _) = evaluate_baseline(model, data, dev, args.dt, args.eps)
        else:
            m, (Pm, _, Pt, _) = evaluate_model(model, data, dev, args.dt, args.eps)
        sep_m = (Pm[:, :, 0] - Pm[:, :, 1]).norm(dim=-1)
        sep_t = (Pt[:, :, 0] - Pt[:, :, 1]).norm(dim=-1)
        d = r * 0.5
        out[str(r)] = {"d": d, "err": [m["err"][i] for i in (4, 9, 19, 49, 99)],
                       "sep_rel_err": [float(((sep_m[:, i] - sep_t[:, i]).abs() / d).mean()) for i in (20, 50, 100)],
                       "energy_drift_rel": m["energy_drift_rel_model"][100], "angmom_drift_rel": m["angmom_drift_rel_model"][100],
                       "energy_drift_rel_true": m["energy_drift_rel_true"][100]}
    return out


def evaluate_baseline(dyn, data, device, dt, eps):
    P, V, M, mask = sf.to_tensors(data, device)
    T = P.shape[1]
    Pm, Vm = torch.zeros_like(P), torch.zeros_like(V)
    for s in range(P.shape[0]):
        n = int(mask[s].sum())
        p0 = (P[s, 0, :n] + sf.CENTER).float()
        v0 = V[s, 0, :n].float()
        with torch.no_grad():
            ps, vs = unroll(dyn, p0, v0, T - 1, dt)
        Pm[s, 0, :n], Vm[s, 0, :n] = P[s, 0, :n], V[s, 0, :n]
        Pm[s, 1:, :n] = ps.double() - sf.CENTER
        Vm[s, 1:, :n] = vs.double()
    out = sf.traj_metrics(Pm, Vm, P, V, M, mask, eps)
    out["const_vel_err"] = sf.const_vel_err(P, V, mask, dt)
    return out, (Pm, Vm, P, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(EXPS), required=True)
    ap.add_argument("--variant", default="all")
    ap.add_argument("--mode", choices=["train", "baseline", "summarize"], default="train")
    ap.add_argument("--opt", default="", help="speed flags: cl, graphtrain, half, bf16 (see scatter_field.apply_opt)")
    ap.add_argument("--baseline-ckpt", default=None)
    ap.add_argument("--init-ckpt", default=None, help="warm-start model weights (optimizer and iteration reset)")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--orbit-mix", type=float, default=0.0)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--eval-scenes", type=int, default=48)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out-dir", default="results/scatter_field")
    ap.add_argument("--ckpt-dir", default="checkpoints/scatter_field")
    args = ap.parse_args()
    cfg = EXPS[args.exp]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    dd = dev if dev.type == "cuda" else None

    def log(s):
        print(s, flush=True)

    if args.mode == "summarize":
        summarize(args, cfg)
        return

    if args.mode == "baseline":
        dyn = CentralForceDynamics(dt=args.dt)
        dyn.load_state_dict(torch.load(args.baseline_ckpt, map_location=dev))
        dyn.to(dev)
        res = {}
        for seed in EVAL_SEEDS:
            data = make_data(cfg, args.eval_scenes, args.eval_steps, seed, dd, args.dt, args.eps)
            res[str(seed)], (Pm, _, Pt, _) = evaluate_baseline(dyn, data, dev, args.dt, args.eps)
            log(f"baseline {args.exp} {seed} err@5/10/20 {headline(res[str(seed)])}")
            if seed == EVAL_SEEDS[0]:
                np.savez(f"{args.out_dir}/{args.exp}_baseline_traj.npz", model=Pm[:4].cpu().numpy(), truth=Pt[:4].cpu().numpy())
        if cfg.get("orbit") or args.exp == "A":
            res["sweep"] = sweep(dyn, dev, args, baseline=True)
        json.dump(res, open(f"{args.out_dir}/{args.exp}_baseline.json", "w"))
        return

    train_data = make_data(cfg, args.train, args.steps, args.seed, dd, args.dt, args.eps)
    if args.orbit_mix > 0:
        k = int(args.train * args.orbit_mix)
        train_data = train_data[k:] + sf.make_orbit_dataset(k, args.steps, args.seed, dd, dt=args.dt, eps=args.eps)
    tensors = sf.to_tensors(train_data, dev)
    names = cfg["variants"] if args.variant == "all" else args.variant.split(",")
    eval_sets = {seed: make_data(cfg, args.eval_scenes, args.eval_steps, seed, dd, args.dt, args.eps) for seed in EVAL_SEEDS}
    for name in names:
        out_path = f"{args.out_dir}/{args.exp}_{name}.json"
        if os.path.exists(out_path):
            log(f"skip {name} (done)")
            continue
        model = build(cfg, name, args.dt).to(dev)
        for o in [x for x in args.opt.split(",") if x]:
            sf.apply_opt(model, o)
        if isinstance(model, sf.MultiLevelScatterField):
            r, rc, glob = 3, 3 * model.h0, True
        else:
            r, rc, glob = sf.receptive_field(model)
        log(f"=== {args.exp} {name} params {sum(p.numel() for p in model.parameters())} receptive field {r} cells = {rc:.1f} units global={glob}")
        if args.init_ckpt:
            model.load_state_dict(torch.load(args.init_ckpt, map_location=dev, weights_only=False)["model"])
            log(f"warm start from {args.init_ckpt}")
        n_it, secs = train(model, tensors, args, f"{args.ckpt_dir}/{args.exp}_{name}.pt", log)
        res = {"train_iters": n_it, "train_seconds": secs, "variant": name, "cfg": {**BASE, **VARIANTS[name]}, "rf_cells": r, "rf_units": rc, "rf_global": glob,
               "args": vars(args), "self_force": sf.self_force_probe(model, dev)}
        for seed in EVAL_SEEDS:
            res[str(seed)], (Pm, _, Pt, _) = evaluate_model(model, eval_sets[seed], dev, args.dt, args.eps)
            log(f"{name} {seed} err@5/10/20 {headline(res[str(seed)])} constvel {[round(res[str(seed)]['const_vel_err'][i], 4) for i in (4, 9, 19)]}")
            if seed == EVAL_SEEDS[0]:
                np.savez(f"{args.out_dir}/{args.exp}_{name}_traj.npz", model=Pm[:4].cpu().numpy(), truth=Pt[:4].cpu().numpy())
        res["curl"] = sf.curl_probe(model, dev)
        log(f"{name} curl/total {res['curl']['curl_over_total']:.3f} self-force {res['self_force']:.4f}")
        if cfg.get("orbit"):
            res["sweep"] = sweep(model, dev, args)
        if cfg["star"]:
            res["probe"] = sf.delete_star_probe(model, dev)
            log(f"{name} delete-star lag steps {res['probe']['lag_steps']} ratio_before {[round(x, 2) for x in res['probe']['ratio_before']]}")
        json.dump(res, open(out_path, "w"))
        del model
        torch.cuda.empty_cache()


def summarize(args, cfg):
    rows = []
    for name in ["baseline"] + cfg["variants"]:
        path = f"{args.out_dir}/{args.exp}_{name}.json"
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        line = [name]
        for seed in EVAL_SEEDS:
            m = r[str(seed)]
            line += [f"{m['err'][i]:.4f}" for i in (4, 9, 19)]
        m = r[str(EVAL_SEEDS[0])]
        line += [f"{m['err'][49]:.3f}", f"{m['err'][99]:.3f}", f"{m['energy_drift_rel_model'][99]:.3f}",
                 f"{m['mom_drift_model'][99]:.2e}", f"{m['angmom_drift_rel_model'][99]:.3f}"]
        line += [f"{r['rf_units']:.0f}" if "rf_units" in r else "-"]
        line += [f"{r['self_force']:.3f}" if "self_force" in r else "-"]
        if "probe" in r:
            line += [str(r["probe"]["lag_steps"])]
        rows.append(line)
    hdr = ["variant", "9000 @5", "@10", "@20", "12000 @5", "@10", "@20", "@50", "@100", "dE/E@100", "dP@100", "dL/L@100", "RF units", "self-f", "lag"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    ref = json.load(open(f"{args.out_dir}/{args.exp}_{[n for n in cfg['variants'] if os.path.exists(f'{args.out_dir}/{args.exp}_{n}.json')][0]}.json"))[str(EVAL_SEEDS[0])]
    print("const-vel err@5/10/20 (9000):", [round(ref["const_vel_err"][i], 4) for i in (4, 9, 19)])
    print("truth dE/E@100", round(ref["energy_drift_rel_true"][99], 4), "truth dP@100", ref["mom_drift_true"][99])


if __name__ == "__main__":
    main()
