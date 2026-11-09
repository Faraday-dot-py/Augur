"""Single-law (inverse-square gravity) grid-network sizing: train MultiScatterField with UNet width 4/8/16/32/64 (levels 5), no law conditioning,
from scratch, then score on held-out scenes. Pre-registration: docs/debugging/experiment-log.md 2026-11-08 single-law-inverse-square-network-sizing.
Recipe (loss, k-ramp, lr schedule, clip-1.0, skip-nonfinite, per-sample drop) is train_scatter_multi.py (branch near-field-retrain 173ed13) with gravity pow2 only,
amplitude 1, constant zero law descriptor. Training seed 4738; held-out evaluation scenes from seeds 4739 and 4740 (never used for training or selection; final
iterate is scored). Per-width full-state checkpoint every --ckpt-every iterations, --resume implicit (finished widths skipped), SIGTERM saves and exits 143.
Usage: PYTHONPATH=. python scripts/single_law_network_sizing.py --out-dir results/single_law_network_sizing --ckpt-dir ckpt
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_multi as sm

STOP = {"flag": False}
LEVELS = 5
REF = 32
EVAL_SEEDS = (4739, 4740)
DOWN_LINE, UP_LINE = 1.10, 0.90
ONE_STEP_TICKS = (0, 10, 20, 30, 40)
ROLL_TICKS = (5, 10, 20)


def on_term(*_):
    STOP["flag"] = True


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def build_model(width, seed, dev):
    torch.manual_seed(seed)
    model = sm.build()
    model.net = sf.UNet(sm.CIN, 2, width, LEVELS)
    torch.nn.init.zeros_(model.net.out.weight)
    torch.nn.init.zeros_(model.net.out.bias)
    torch.manual_seed(seed)
    k, p = sm.CondMLP(4, 128), sm.CondMLP(4, 128)
    model.kmlp.load_state_dict(k.state_dict())
    model.ppmlp.mlp.load_state_dict(p.state_dict())
    model = model.to(dev)
    model.set_law(torch.zeros(1, sm.DC, device=dev), None)
    return model


def count_params(model):
    n = lambda m: sum(p.numel() for p in m.parameters())
    return {"unet": n(model.net), "kernel_mlp": n(model.kmlp), "pair_mlp": n(model.ppmlp), "total": n(model)}


def gen_data(seed, tag, scenes, steps, dev):
    return sm.gen_cases([("pow2", "single", 1.0)] * scenes, seed, dev, steps, tag=tag)


def save(path, model, opt, it, args, extra):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "args": vars(args), **extra}, tmp)
    os.replace(tmp, path)


def train(width, args, dev, data):
    path = f"{args.ckpt_dir}/width{width}.pt"
    model = build_model(width, args.seed, dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    iters = args.epochs * (args.scenes // args.batch)
    it0, elapsed0, kgate, losses, skipped, dropped = 0, 0.0, args.k_start, [], 0, 0
    if os.path.exists(path):
        st = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        it0, elapsed0, kgate = st["it"], st["elapsed"], st["kgate"]
        losses, skipped, dropped = st["epoch_loss"], st["skipped"], st["dropped"]
        log(f"width {width} resumed at it {it0}/{iters} (elapsed {elapsed0:.0f}s)")
    log(f"width {width} params {count_params(model)}")
    S = data["P"].shape[0]
    nsc = data["n"].to(dev)
    per_epoch = args.scenes // args.batch
    t_start = time.time()
    it = it0
    run_loss = torch.zeros((), device=dev)
    dropped_prev = dropped
    while it < iters:
        frac = it / max(1, iters - 1)
        k = min(int(round(args.k_start + (args.k_end - args.k_start) * min(1.0, frac / args.k_ramp))), kgate)
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        g = torch.Generator(device="cpu").manual_seed(args.seed * 1000003 + it)
        idx = torch.randint(S, (args.batch,), generator=g).to(dev)
        t0 = torch.randint(0, args.steps - k + 1, (args.batch,), generator=g).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]
        Nb = int(nsc[idx].max())
        Pw = data["P"][idx[:, None], tt][:, :, :Nb]
        Vw = data["V"][idx[:, None], tt][:, :, :Nb]
        mk = data["mask"][idx][:, :Nb]
        denom = mk.sum() * 2 * k
        opt.zero_grad()
        tot, kept = 0.0, 0.0
        for c0 in range(0, args.batch, args.micro):
            sl = slice(c0, c0 + args.micro)
            m = torch.ones_like(mk[sl])
            pos, vel = Pw[sl, 0], Vw[sl, 0]
            field = model.init_field(m.shape[0], dev)
            w = mk[sl][:, :, None]
            eps_l, evs_l = [], []
            for s in range(k):
                pos, vel, _, field, _ = sf._step_m(model, pos, vel, m, mk[sl], field)
                pos, vel = pos.clamp(-1e3, 1e3), vel.clamp(-1e3, 1e3)
                dd = (pos - Pw[sl, s + 1]).norm(dim=-1, keepdim=True)
                hub = torch.where(dd <= args.huber, dd ** 2, args.huber * (2 * dd - args.huber))
                eps_l.append((hub * w).sum((1, 2)) / denom)
                evs_l.append((((vel - Vw[sl, s + 1]) ** 2) * w).sum((1, 2)) / denom)
            per_g = torch.stack(eps_l).sum(0)
            per = per_g.detach()
            keep = ((per * args.batch < args.sample_cap) & torch.isfinite(per)).float()
            dropped += int(keep.numel() - keep.sum())
            loss = (torch.nan_to_num(per_g + 0.1 * torch.stack(evs_l).sum(0)) * keep).sum()
            loss.backward()
            tot = tot + loss.detach()
            kept += float(keep.sum())
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(tot) and torch.isfinite(gn) and kept >= 1:
            opt.step()
            run_loss += tot
        else:
            opt.zero_grad()
            skipped += 1
            if skipped % 50 == 1:
                log(f"width {width} it {it} loss {float(tot):.3g} non-finite or above cap, step skipped ({skipped} total)")
        it += 1
        if it % args.log_every == 0:
            log(f"width {width} it {it}/{iters} k {k} lr {lr:.2e} loss {float(tot):.6f} dropped {dropped} {(time.time() - t_start) / (it - it0):.3f}s/it")
        if it % args.gate_every == 0:
            wd = (dropped - dropped_prev) / (args.gate_every * args.batch)
            dropped_prev = dropped
            if wd < 0.05:
                kgate = min(args.k_end, kgate + 1)
            elif wd > 0.25:
                kgate = max(args.k_start, kgate - 2)
            log(f"width {width} gate: window drop rate {wd:.3f} -> kgate {kgate}")
        if it % per_epoch == 0:
            losses.append(float(run_loss) / per_epoch)
            run_loss.zero_()
        stop = STOP["flag"]
        if stop or it % args.ckpt_every == 0 or it == iters:
            if all(torch.isfinite(p).all() for p in model.parameters()):
                save(path, model, opt, it, args, {"elapsed": elapsed0 + time.time() - t_start, "kgate": kgate, "epoch_loss": losses,
                                                  "skipped": skipped, "dropped": dropped})
            else:
                log(f"width {width} non-finite parameters, checkpoint NOT written")
        if stop:
            log(f"width {width} SIGTERM: saved at it {it}, exiting 143")
            sys.exit(143)
    elapsed = elapsed0 + time.time() - t_start
    return model, {"iters": iters, "seconds": elapsed, "s_per_it_this_run": (time.time() - t_start) / max(1, it - it0), "skipped_steps": skipped,
                   "dropped_samples": dropped, "epoch_loss": losses, "kgate_final": kgate}


@torch.no_grad()
def evaluate(model, data, dev, steps):
    S = data["P"].shape[0]
    nf = min(40, steps)
    nr = min(20, steps)
    ones_tick = [t for t in ONE_STEP_TICKS if t + 1 <= steps]
    rt = sorted({min(t, nr) for t in ROLL_TICKS})
    fe = torch.zeros(2, dtype=torch.float64)
    ft = torch.zeros(2, dtype=torch.float64)
    fl = torch.zeros(2, dtype=torch.float64)
    fcount = 0.0
    one_se = torch.zeros(4, dtype=torch.float64)
    one_n = 0
    roll_err = {t: [] for t in rt}
    roll_se = torch.zeros(4, dtype=torch.float64)
    roll_n = 0
    force_scene = []
    for c0 in range(0, S, 16):
        sl = slice(c0, min(S, c0 + 16))
        Nb = int(data["n"][sl].max())
        P, V = data["P"][sl][:, :, :Nb], data["V"][sl][:, :, :Nb]
        mk = data["mask"][sl][:, :Nb]
        par = data["params"][sl]
        B = P.shape[0]
        m = torch.ones_like(mk)
        w = mk[:, :, None].double()
        f = model.init_field(B, dev)[0]
        scene_e = torch.zeros(B, dtype=torch.float64, device=dev)
        scene_t = torch.zeros(B, dtype=torch.float64, device=dev)
        for t in range(nf + 1):
            a, f = model.force(P[:, t], V[:, t], m, mk, f)
            if t == 0:
                continue
            ref = sm.exact_acc(P[:, t].double(), None, mk.double(), par)
            d = (a.double() - ref) * w
            fe += (d ** 2).sum((0, 1)).cpu()
            ft += ((ref * w) ** 2).sum((0, 1)).cpu()
            scene_e += (d ** 2).sum((1, 2))
            scene_t += ((ref * w) ** 2).sum((1, 2))
            r32 = sm.exact_acc(P[:, t], None, mk, par.float()).double()
            fl += (((r32 - ref) * w) ** 2).sum((0, 1)).cpu()
            fcount += float(mk.sum())
        force_scene.append((scene_e / scene_t).sqrt().cpu())
        for t in ones_tick:
            fld = model.init_field(B, dev)
            pos, vel, _, _, _ = sf._step_m(model, P[:, t], V[:, t], m, mk, fld)
            e = torch.cat([pos - P[:, t + 1], vel - V[:, t + 1]], -1).double() * w
            one_se += (e ** 2).sum((0, 1)).cpu()
            one_n += float(mk.sum())
        pos, vel = P[:, 0], V[:, 0]
        fld = model.init_field(B, dev)
        for s in range(nr):
            pos, vel, _, fld, _ = sf._step_m(model, pos, vel, m, mk, fld)
            if s + 1 in roll_err:
                d = (pos - P[:, s + 1]).norm(dim=-1) * mk
                roll_err[s + 1].append((d.sum(1) / mk.sum(1)).cpu())
        e = torch.cat([pos - P[:, nr], vel - V[:, nr]], -1).double() * w
        roll_se += (e ** 2).sum((0, 1)).cpu()
        roll_n += float(mk.sum())
    ch = ["pos_x", "pos_y", "vel_x", "vel_y"]
    out = {
        "force_rel_l2": float((fe.sum() / ft.sum()).sqrt()),
        "force_rel_l2_by_channel": {"ax": float((fe[0] / ft[0]).sqrt()), "ay": float((fe[1] / ft[1]).sqrt())},
        "force_fp32_exact_floor_rel_l2": float((fl.sum() / ft.sum()).sqrt()),
        "force_scene_median": float(torch.cat(force_scene).median()),
        "force_scene_max": float(torch.cat(force_scene).max()),
        "force_ticks": nf,
        "one_step_rmse_by_channel": {c: float((one_se[i] / one_n).sqrt()) for i, c in enumerate(ch)},
        "one_step_ticks": ones_tick,
        "one_step_mass_err": 0.0,
        "rollout_mean_pos_err": {str(t): float(torch.cat(roll_err[t]).mean()) for t in rt},
        "rollout_median_pos_err": {str(t): float(torch.cat(roll_err[t]).median()) for t in rt},
        "rollout_max_scene_pos_err": {str(t): float(torch.cat(roll_err[t]).max()) for t in rt},
        "rollout_rmse_by_channel_at_end": {c: float((roll_se[i] / roll_n).sqrt()) for i, c in enumerate(ch)},
        "rollout_end_tick": nr,
        "scenes": S,
    }
    out["force_abs_rms_by_channel"] = {"ax": float((fe[0] / fcount).sqrt()), "ay": float((fe[1] / fcount).sqrt())}
    return out


def verdict(res, widths):
    f = {w: [res[w]["eval"][str(s)]["force_rel_l2"] for s in EVAL_SEEDS] for w in widths}
    ref = f[REF]
    out = {"per_width": {}, "line_down": DOWN_LINE, "line_up": UP_LINE, "reference_width": REF}
    for w in widths:
        per_seed = [a / b for a, b in zip(f[w], ref)]
        out["per_width"][str(w)] = {"force_rel_l2_by_seed": f[w], "mean_force_rel_l2": float(np.mean(f[w])),
                                    "ratio_of_means_vs_32": float(np.mean(f[w]) / np.mean(ref)), "ratio_by_seed_vs_32": per_seed}

    def status(r, line, below):
        ok = [(x <= line) if not below else (x < line) for x in r]
        return "QUALIFY" if all(ok) else "FAIL" if not any(ok) else "SPLIT"

    down = {}
    for w in [x for x in widths if x < REF]:
        pw = out["per_width"][str(w)]
        s = status(pw["ratio_by_seed_vs_32"], DOWN_LINE, False)
        down[str(w)] = s if s != "QUALIFY" or pw["ratio_of_means_vs_32"] <= DOWN_LINE else "FAIL"
    out["downscale_status"] = down
    ans = None
    for w in sorted(int(x) for x in down):
        if down[str(w)] == "QUALIFY":
            ans = ("PASS", w)
            break
        if down[str(w)] == "SPLIT":
            ans = ("INCONCLUSIVE", w)
            break
    up = None
    if max(widths) > REF:
        pw = out["per_width"][str(max(widths))]
        up = status(pw["ratio_by_seed_vs_32"], UP_LINE, True)
        if up == "QUALIFY" and pw["ratio_of_means_vs_32"] >= UP_LINE:
            up = "FAIL"
    out["upscale_status"] = up
    if ans is not None:
        out["label"], out["smallest_width"] = ans
    elif up == "QUALIFY":
        out["label"], out["smallest_width"] = "UPSCALE", None
    elif up == "SPLIT":
        out["label"], out["smallest_width"] = "INCONCLUSIVE", None
    else:
        out["label"], out["smallest_width"] = "KILL", None
    out["also_capacity_limited"] = up == "QUALIFY"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--scenes", type=int, default=2560)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--k-start", type=int, default=2)
    ap.add_argument("--k-end", type=int, default=30)
    ap.add_argument("--k-ramp", type=float, default=0.8)
    ap.add_argument("--huber", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--sample-cap", type=float, default=100.0)
    ap.add_argument("--gate-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-scenes", type=int, default=64)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    widths = [int(x) for x in args.widths.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term)
    t0 = time.time()
    train_data = gen_data(args.seed, 1, args.scenes, args.steps, dev)
    assert train_data["P"].shape[0] == args.scenes, "training scene generation rejected scenes"
    evals = {s: gen_data(s, 2, args.eval_scenes, args.steps, dev) for s in EVAL_SEEDS}
    for s, d in evals.items():
        assert d["P"].shape[0] == args.eval_scenes, f"eval seed {s} short"
    log(f"data {args.scenes} train + {args.eval_scenes} x {len(EVAL_SEEDS)} held-out scenes in {time.time() - t0:.0f}s")
    res = {}
    for w in widths:
        rp = f"{args.out_dir}/width{w}.json"
        if os.path.exists(rp):
            res[w] = json.load(open(rp))
            log(f"width {w} result present, skipped")
            continue
        model, info = train(w, args, dev, train_data)
        model.set_law(torch.zeros(1, sm.DC, device=dev), None)
        ev = {str(s): evaluate(model, d, dev, args.steps) for s, d in evals.items()}
        res[w] = {"width": w, "levels": LEVELS, "params": count_params(model), "train": info, "eval": ev,
                  "train_seed": args.seed, "eval_seeds": list(EVAL_SEEDS), "args": vars(args)}
        json.dump(res[w], open(rp + ".tmp", "w"), indent=1)
        os.replace(rp + ".tmp", rp)
        log(f"width {w} done: force_rel_l2 " + " ".join(f"{s}={ev[str(s)]['force_rel_l2']:.5f}" for s in EVAL_SEEDS))
    if REF in res:
        v = verdict(res, [w for w in widths if w in res])
        json.dump(v, open(f"{args.out_dir}/verdict.json", "w"), indent=1)
        log("verdict " + json.dumps({k: v[k] for k in ("label", "smallest_width", "downscale_status", "upscale_status")}))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
