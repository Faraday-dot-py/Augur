"""Fine-tune a scatter-field checkpoint on solar-scale scenes (star + 8 planets + moon + 200 belt particles, 2000 ticks) with a longer unroll.

Arms (one run each, separate checkpoint):
  solar-scale-same-horizon-control   solar data, unroll k 4 -> 30 (the orbit_ft2 horizon)
  solar-scale-long-horizon           solar data, unroll k 10 -> 100
Both train with the kernel-potential edge fix (--kpad ext) so training matches the evaluation configuration.
Loss as in scripts/train_scatter_field.py: squared position error + 0.1 x squared velocity error, averaged over bodies and unroll steps.
Resumable: full-state checkpoint (model, optimizer, iteration) every --ckpt-every iterations, written atomically; SIGTERM saves and exits 0; the
schedule is a function of the iteration so a resumed run is the same run.

  python scripts/train_scatter_solar_stability.py --arm solar-scale-long-horizon --init-ckpt orbit_ft2.pt --k-start 10 --k-end 100
"""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import scatter_solar as sol
from scripts import train_scatter_field as tsf

VAR = "ms_kp_pot_v_g128"
TRAIN_SEED_BASE = 100000
HILL_K = 4.0
STOP = [False]


def build_model(init_ckpt, dev, kpad):
    model = tsf.build(tsf.EXPS["E"], VAR, 0.1).to(dev)
    sf.load_compat(model, torch.load(init_ckpt, map_location=dev, weights_only=False)["model"])
    model.kpad = kpad
    return model


def make_scenes(num, steps, dev, belt=200, log=print):
    """Solar-scale training scenes: ic seeds TRAIN_SEED_BASE + i (disjoint from the held-out eval seeds), exact fp64 truth on the GPU. Returns centered float32 tensors on dev."""
    P, V, M = [], [], []
    t0 = time.time()
    for i in range(num):
        sc = sol.solar_scene(TRAIN_SEED_BASE + i, belt, HILL_K)
        Pt, Vt = ss.nbody_truth(sc["pos"], sc["vel"], sc["mass"], steps, dev)
        P.append(torch.tensor(Pt, dtype=torch.float32, device=dev))
        V.append(torch.tensor(Vt, dtype=torch.float32, device=dev))
        M.append(torch.tensor(sc["mass"], dtype=torch.float32, device=dev))
        if i % 8 == 0:
            log(f"scene {i}/{num} truth {time.time() - t0:.0f}s")
    return torch.stack(P), torch.stack(V), torch.stack(M)


def k_at(it, iters, k_start, k_end):
    return int(round(k_start + (k_end - k_start) * it / max(1, iters - 1)))


def staircase_at(it, iters, k_start, k_end, stages):
    """n-ramp: `stages` equal blocks of iterations. Block s: first half raises k to k_s with the previous belt fraction, second half raises the belt fraction to s/(stages-1). Block 0 has no belt; the last block ends at k_end with the full belt."""
    s = min(stages - 1, it * stages // iters)
    second = (it * stages / iters - s) >= 0.5
    k = int(round(k_start + (k_end - k_start) * s / (stages - 1)))
    return k, (s if second else max(0, s - 1)) / (stages - 1)


def staircase_schedule(iters, k_start, k_end, stages):
    """[(first iteration, k, belt fraction)] for every change."""
    out = []
    for it in range(iters):
        cur = staircase_at(it, iters, k_start, k_end, stages)
        if not out or out[-1][1:] != cur:
            out.append((it, *cur))
    return out


RAMP_ORDER = ("k", "n", "m")
M_FLOOR = 0.25


def parse_ramp(spec):
    """None = no regime logic (old behaviour); "none" = every factor at full difficulty from iteration 0; else a comma list of the ramped factors from k (rollout steps), n (belt bodies), m (mass-range admission)."""
    if spec is None:
        return None
    if spec == "none":
        return ()
    f = tuple(spec.split(","))
    assert all(x in RAMP_ORDER for x in f) and len(set(f)) == len(f), spec
    return tuple(x for x in RAMP_ORDER if x in f)


def ramp_levels(it, iters, ramp, stages):
    """Level in [0, 1] of each ramped factor at iteration it. `stages` equal blocks; block s raises the factors one at a time, in the order k, n, m, each in its own equal sub-phase (factor j reaches s/(stages-1) at sub-phase j, until then it holds the previous block's level). With two factors k, n this is staircase_at."""
    F = len(ramp)
    s = min(stages - 1, it * stages // iters)
    p = min(F - 1, int((it * stages / iters - s) * F))
    return {f: (s if p >= j else max(0, s - 1)) / (stages - 1) for j, f in enumerate(ramp)}


def regime_at(it, iters, ramp, stages, k_start, k_end, n_scenes):
    """(k, belt fraction, number of admitted scenes) for the regime. A factor not in ramp is at full difficulty (k_end, 1.0, all scenes). k alone is the continuous linear ramp of the existing recipe."""
    if ramp == ("k",):
        return k_at(it, iters, k_start, k_end), 1.0, n_scenes
    lv = ramp_levels(it, iters, ramp, stages) if ramp else {}
    lk, ln, lm = (lv.get(f, 1.0) for f in RAMP_ORDER)
    return int(round(k_start + (k_end - k_start) * lk)), ln, int(np.ceil(n_scenes * (M_FLOOR + (1 - M_FLOOR) * lm) - 1e-9))


def mass_rank(M):
    """Scene order by mass-range width: log of max / min positive mass over the real bodies, narrowest first (stable)."""
    mx = M.amax(1)
    mn = torch.where(M > 0, M, torch.full_like(M, float("inf"))).amin(1)
    return torch.argsort(torch.log(mx / mn), stable=True)


def lr_at(it, iters, lr, warmup):
    frac = it / max(1, iters - 1)
    return lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac))) * min(1.0, (it + 1) / warmup)


def translation_offsets(Pw, g, amp, limit):
    """Random whole-scene shift per window (B, 2), uniform in [-amp, amp] per axis intersected with the range that keeps every body of the window inside |x| <= limit (the grid covers +-extent/2); 0 if the window already exceeds limit."""
    lo = torch.maximum(-amp * torch.ones_like(Pw[:, 0, 0, :]), -limit - Pw.amin(dim=(1, 2)))
    hi = torch.minimum(amp * torch.ones_like(lo), limit - Pw.amax(dim=(1, 2)))
    ok = (hi >= lo).to(Pw.dtype)
    u = torch.rand(Pw.shape[0], 2, generator=g).to(Pw.device)
    return ok * (lo + u * (hi - lo).clamp_min(0))


def save_state(path, model, opt, it, extra):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, **extra}, tmp)
    os.replace(tmp, path)


def train(model, data, args, ckpt_path, log):
    P, V, M = data[:3]
    K, HOST, BRANK, BN = data[3:7] if len(data) > 3 else (None,) * 4
    mode = getattr(args, "loss_mode", "default")
    ramp = parse_ramp(getattr(args, "ramp", None))
    order = mass_rank(M) if ramp and "m" in ramp else None
    S, T, N = P.shape[0], P.shape[1], P.shape[2]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, skipped, hist = 0, 0, []
    if os.path.exists(ckpt_path) and not args.fresh:
        st = torch.load(ckpt_path, map_location=P.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, skipped = st["it"], st.get("skipped", 0)
        log(f"resumed from it {start}")
    fns = sf.make_step_fns(model, args.batch, N, args.k_end, P.device) if "graphtrain" in getattr(model, "opt", ()) else None
    g = torch.Generator(device="cpu")
    t_start = time.time()
    it = start
    while it < args.iters and not STOP[0]:
        g.manual_seed(args.seed * 1000003 + it)
        nadm = S
        if ramp is not None:
            k, bfrac, nadm = regime_at(it, args.iters, ramp, getattr(args, "curriculum_stages", 0) or 12, args.k_start, args.k_end, S)
        elif getattr(args, "curriculum_stages", 0) > 0:
            k, bfrac = staircase_at(it, args.iters, args.k_start, args.k_end, args.curriculum_stages)
        else:
            k, bfrac = k_at(it, args.iters, args.k_start, args.k_end), 1.0
        for pg in opt.param_groups:
            pg["lr"] = lr_at(it, args.iters, args.lr, args.warmup)
        idx = torch.randint(nadm, (args.batch,), generator=g).to(P.device)
        if order is not None:
            idx = order[idx]
        t0 = torch.randint(0, T - 1 - k + 1, (args.batch,), generator=g).to(P.device)
        tt = t0[:, None] + torch.arange(k + 1, device=P.device)[None]
        Pw, Vw = P[idx[:, None], tt], V[idx[:, None], tt]
        m = M[idx]
        if getattr(args, "translate_amp", 0.0) > 0:
            Pw = Pw + translation_offsets(Pw, g, args.translate_amp, args.translate_limit)[:, None, None, :]
        mask = (m > 0).to(P.dtype)
        if ramp is None and getattr(args, "curriculum_stages", 0) > 0 or ramp and "n" in ramp:
            keep = (K[idx] != 3) | (BRANK[idx] < torch.ceil(bfrac * BN[idx])[:, None])
            mask = mask * keep.to(P.dtype)
            m = m * mask
        pos, vel = Pw[:, 0], Vw[:, 0]
        field = model.init_field(args.batch, P.device)
        lp = lv = 0.0
        wb = mask * (1 + (args.moon_weight - 1) * (K[idx] == 2).to(P.dtype)) if mode == "moon-weighted" else mask
        w = wb[:, :, None]
        denom = wb.sum() * 2 * k
        if mode == "relative-error":
            kk, hh = K[idx], HOST[idx]
            p0 = Pw[:, 0]
            st = ((kk == 0) | (kk == 4)).to(P.dtype)
            c = (p0 * (M[idx] * st)[:, :, None]).sum(1) / (M[idx] * st).sum(1)[:, None].clamp_min(1e-12)
            r_star = (p0 - c[:, None]).norm(dim=-1)
            r_host = (p0 - torch.gather(p0, 1, hh.clamp_min(0)[:, :, None].expand(-1, -1, 2))).norm(dim=-1)
            r = torch.where(kk == 2, r_host, r_star).clamp_min(0.1)
            r2n = ((mask * r ** 2).sum() / mask.sum()).detach()
            r2 = torch.where(st > 0, r2n, r ** 2)
            pscale = (r2n / r2)[:, :, None]
        else:
            pscale = 1.0
        for s in range(k):
            pos, vel, m, field, _ = fns[s](pos, vel, m, mask, field) if fns else sf._step_m(model, pos, vel, m, mask, field)
            lp = lp + (((pos - Pw[:, s + 1]) ** 2) * w * pscale).sum() / denom
            lv = lv + (((vel - Vw[:, s + 1]) ** 2) * w).sum() / denom
        loss = lp + 0.1 * lv
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lval = loss.item()
        spike = args.spike_factor > 0 and len(hist) >= 20 and lval > args.spike_factor * float(np.median(hist[-50:]))
        if np.isfinite(lval):
            hist.append(lval)
        if torch.isfinite(loss) and torch.isfinite(gn) and not spike:
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} loss {lval:.4g} non-finite or spike, step skipped ({skipped} total)")
        it += 1
        if it % args.log_every == 0:
            log(f"it {it}/{args.iters} k {k} belt {bfrac:.3f} scenes {nadm} loss {lval:.6f} lp {lp.item():.6f} lv {lv.item():.6f} gn {float(gn):.3f} t {time.time() - t_start:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30 if P.is_cuda else 0:.1f}GB")
        if getattr(args, "snapshot_every", 0) and it % args.snapshot_every == 0 and all(torch.isfinite(p).all() for p in model.parameters()):
            torch.save({"model": model.state_dict(), "it": it}, ckpt_path[:-3] + f"_it{it}.pt")
        if it % args.ckpt_every == 0 and all(torch.isfinite(p).all() for p in model.parameters()):
            save_state(ckpt_path, model, opt, it, {"skipped": skipped})
    if all(torch.isfinite(p).all() for p in model.parameters()):
        save_state(ckpt_path, model, opt, it, {"skipped": skipped})
    log(f"stopped at it {it} of {args.iters} (signal={STOP[0]}), {time.time() - t_start:.0f}s, skipped {skipped}")
    return it, skipped, time.time() - t_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--scenes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--epoch-iters", type=int, default=120, help="iterations per epoch (random-window sampling has no natural epoch)")
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--spike-factor", type=float, default=0.0, help="skip a step whose loss exceeds this x the recent median (0 = off)")
    ap.add_argument("--kpad", choices=["zero", "ext"], default="ext")
    ap.add_argument("--translate-amp", type=float, default=0.0, help="random whole-scene translation per training window, uniform in [-amp, amp] per axis (0 = off)")
    ap.add_argument("--translate-limit", type=float, default=30.0, help="shifted windows keep every body inside |x| <= limit (grid half-width 32)")
    ap.add_argument("--opt", default="default", help="speed flags (scatter_field.resolve_opt); default = cl,graphtrain")
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/solar_stability")
    ap.add_argument("--ckpt-dir", default="ckpt/solar_stability")
    args = ap.parse_args()
    args.iters = args.epochs * args.epoch_iters
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__(0, True))

    def log(s):
        print(s, flush=True)

    model = build_model(args.init_ckpt, dev, args.kpad)
    sf.apply_opts(model, args.opt, "train")
    model.kpad = args.kpad
    model.train()
    log(f"=== {args.arm} params {sum(p.numel() for p in model.parameters())} init {args.init_ckpt} iters {args.iters} k {args.k_start}->{args.k_end} batch {args.batch} kpad {model.kpad} opt {sorted(model.opt)}")
    data = make_scenes(args.scenes, args.steps, dev, log=log)
    ckpt = f"{args.ckpt_dir}/{args.arm}.pt"
    it, skipped, secs = train(model, data, args, ckpt, log)
    if it >= args.iters:
        json.dump({"arm": args.arm, "iters": it, "skipped": skipped, "seconds": secs, "ckpt": ckpt, "args": vars(args)}, open(f"{args.out_dir}/{args.arm}_train.json", "w"), indent=1)
        log(f"DONE {args.arm}")


if __name__ == "__main__":
    main()
