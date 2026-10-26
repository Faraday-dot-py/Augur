"""Train/finetune the Exp E scatter field on small periodic LJ boxes (docs/debugging/scatter-lj.md).

Truth: lj_sim velocity Verlet (dt 0.005) recorded every SUB = 5 steps = one model step (0.025 tau). Training boxes are one tile (model side
11-18), densities log-uniform 0.05-1.05 with the band 0.62-0.82 held out (as scripts/train_lj.py), T uniform 0.2-2.5 plus a hot/dense fraction.
Free rollout of k steps (curriculum k_start -> k_end), loss 100 * |wrapped pos error|^2 + |vel error|^2 in LJ units (as train_lj.py).
--resume restarts from the last checkpoint (written atomically every --ckpt-every iterations and on SIGTERM); data is cached next to it.
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

from scripts import lj_sim
from scripts import scatter_lj as sl

HELD_OUT_RHO = (0.62, 0.82)
STOP = []


def sample_state(rng, tmax, hot_frac):
    if rng.random() < hot_frac:
        return float(rng.uniform(0.9, 1.05)), float(rng.uniform(2.0, 5.0))
    while True:
        rho = float(np.exp(rng.uniform(math.log(0.05), math.log(1.05))))
        if not HELD_OUT_RHO[0] <= rho <= HELD_OUT_RHO[1]:
            break
    return rho, float(rng.uniform(0.2, tmax))


def sample_box(rng, rho):
    while True:
        side = sl.LS * float(rng.uniform(11.0, 18.0))
        n = max(6, int(round(rho * side * side)))
        nx = max(2, int(round(math.sqrt(n / 1.155))))
        ny = max(2, int(round(n / nx)))
        a = math.sqrt(2.0 / (lj_sim.SQ3 * rho))
        box = np.array([nx * a, ny * a * lj_sim.SQ3 / 2])
        if box.min() / sl.LS >= 10.5 and box.max() / sl.LS <= sl.CORE_MAX:
            return nx, ny


def make_data(scenes, frames, warmup, seed, tmax, hot_frac, dev, log):
    rng = np.random.default_rng(seed)
    out = []
    t0 = time.time()
    for i in range(scenes):
        rho, temp = sample_state(rng, tmax, hot_frac)
        nx, ny = sample_box(rng, rho)
        pos, box = lj_sim.lattice(nx, ny, rho, 0.02, rng)
        vel = lj_sim.maxwell(len(pos), temp, rng)
        fp, fv, _ = lj_sim.run(pos, vel, box, warmup + frames * sl.SUB, sl.SUB, 0.005, dev)
        k0 = warmup // sl.SUB
        out.append((fp[k0:], fv[k0:], box, rho, temp))
        if i % 50 == 0:
            log(f"data {i}/{scenes} N {len(pos)} rho {rho:.2f} T {temp:.2f} {time.time() - t0:.0f}s")
    out.sort(key=lambda o: o[0].shape[1])
    nmax = out[-1][0].shape[1]
    F = out[0][0].shape[0]
    P = np.zeros((scenes, F, nmax, 2), np.float32)
    V = np.zeros_like(P)
    for i, (fp, fv, *_r) in enumerate(out):
        P[i, :, :fp.shape[1]], V[i, :, :fp.shape[1]] = fp, fv
    return {"P": P, "V": V, "n": np.array([o[0].shape[1] for o in out]), "box": np.stack([o[2] for o in out]),
            "rho": np.array([o[3] for o in out]), "T": np.array([o[4] for o in out])}


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def to_dev(d, dev):
    mask = (np.arange(d["P"].shape[2])[None, :] < d["n"][:, None]).astype(np.float32)
    return {"P": torch.tensor(d["P"], device=dev), "V": torch.tensor(d["V"], device=dev), "mask": torch.tensor(mask, device=dev),
            "box": torch.tensor(d["box"], device=dev, dtype=torch.float32), "n": d["n"]}


def run_batch(model, pt, D, idx, t0, k):
    ls = sl.LS
    mask, box = D["mask"][idx], D["box"][idx] / ls
    lay = sl.layout(box[0])
    pos, vel = D["P"][idx, t0] / ls, D["V"][idx, t0] / ls
    st = (pt.init_field(len(idx), lay, pos.device), None)
    lp = lv = 0.0
    cnt = mask.sum(1)
    for s in range(k):
        for _ in range(model.nsub):
            pos, vel, st = pt.step(pos, vel, mask, box, lay, st)
        tp, tv = D["P"][idx, t0 + s + 1] / ls, D["V"][idx, t0 + s + 1] / ls
        lp = lp + ((sl.wrap_diff(pos, tp, box) * ls) ** 2).sum(-1).mul(mask).sum(1).div(cnt).mean()
        lv = lv + (((vel - tv) * ls) ** 2).sum(-1).mul(mask).sum(1).div(cnt).mean()
    return lp / k, lv / k, pt.take_pen().to(lp.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init-ckpt", default=None, help="Exp E checkpoint to finetune from (omit for scratch with stock zero-init heads)")
    ap.add_argument("--ckpt-dir", default="checkpoints/scatter_lj")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k-start", type=int, default=8)
    ap.add_argument("--k-end", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scenes", type=int, default=320)
    ap.add_argument("--val-scenes", type=int, default=32)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--tmax", type=float, default=2.5)
    ap.add_argument("--hot-frac", type=float, default=0.15)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--energy", action="store_true", help="mesh force = -autograd grad of a density-only scalar energy (hole-1 fix)")
    ap.add_argument("--resid", action="store_true", help="add a soft-penalised residual force net (momentum channels)")
    ap.add_argument("--vblind", action="store_true", help="zero the velocity-density scatter channels (original non-gradient mesh)")
    ap.add_argument("--norec", action="store_true", help="zero the recurrent field input (original non-gradient mesh)")
    ap.add_argument("--asym", action="store_true", help="pair term without built-in reciprocity (needs --energy); soft reciprocity + momentum penalties")
    ap.add_argument("--substeps", type=int, default=1, help="model steps of dt 0.025/substeps per stored frame")
    ap.add_argument("--pair-sub", type=int, default=1, help="pair-only substeps with cached neighbour list per model step")
    ap.add_argument("--oracle", action="store_true", help="pair term = exact shifted-force LJ (needs --energy); mesh learned")
    ap.add_argument("--nomesh", action="store_true", help="mesh force zeroed (needs --energy); learned pair only")
    ap.add_argument("--lam-recip", type=float, default=10.0)
    ap.add_argument("--lam-mom", type=float, default=100.0)
    ap.add_argument("--lam-mag", type=float, default=300.0)
    ap.add_argument("--lam-curl", type=float, default=300.0)
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    base = os.path.join(args.ckpt_dir, args.tag)
    signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))

    def log(s):
        print(s, flush=True)

    if os.path.exists(base + ".done"):
        log("already done")
        return
    cache = f"{args.ckpt_dir}/data_s{args.seed}_n{args.scenes}_f{args.frames}_t{args.tmax}_h{args.hot_frac}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        tr = {k: z["tr_" + k] for k in ("P", "V", "n", "box")}
        va = {k: z["va_" + k] for k in ("P", "V", "n", "box")}
        log(f"data loaded {cache}")
    else:
        t0 = time.time()
        d = make_data(args.scenes + args.val_scenes, args.frames, args.warmup, args.seed, args.tmax, args.hot_frac, dev, log)
        rs = np.random.default_rng(args.seed + 1).permutation(len(d["n"]))
        vi, ti = np.sort(rs[:args.val_scenes]), np.sort(rs[args.val_scenes:])
        tr = {k: d[k][ti] for k in ("P", "V", "n", "box")}
        va = {k: d[k][vi] for k in ("P", "V", "n", "box")}
        np.savez(cache + ".tmp.npz", **{"tr_" + k: v for k, v in tr.items()}, **{"va_" + k: v for k, v in va.items()})
        os.replace(cache + ".tmp.npz", cache)
        log(f"data generated in {time.time() - t0:.0f}s")
    Dtr, Dva = to_dev(tr, dev), to_dev(va, dev)

    torch.manual_seed(args.seed)
    model = sl.build_model(args.init_ckpt, dev, args.energy, args.resid, args.vblind, args.norec, args.asym, args.oracle, args.nomesh, args.substeps)
    model.nsub = args.substeps
    model.pair_sub, model.pair_cache = args.pair_sub, args.pair_sub > 1
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = 0
    if args.resume and os.path.exists(base + ".pt"):
        st = torch.load(base + ".pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["it"]
        log(f"resumed from it {start}")
    pt = sl.PeriodicTiles(model)
    S = Dtr["P"].shape[0]
    F = Dtr["P"].shape[1]
    hist = []
    skipped = 0
    t_start = time.time()
    it = start

    def save(path):
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "args": vars(args)}, path)

    while it < args.iters:
        frac = it / max(1, args.iters - 1)
        k = int(round(args.k_start + (args.k_end - args.k_start) * frac))
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        g = torch.Generator().manual_seed(args.seed * 100003 + it)
        i0 = int(torch.randint(0, S - args.batch + 1, (1,), generator=g))
        idx = torch.arange(i0, i0 + args.batch, device=dev)
        t0 = torch.randint(0, F - k, (args.batch,), generator=g).to(dev)
        lp, lv, pen = run_batch(model, pt, Dtr, idx, t0, k)
        loss = 100 * lp + lv + args.lam_mag * pen[0] + args.lam_curl * pen[1] + args.lam_recip * pen[2] + args.lam_mom * pen[3]
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
            skipped += 1
            log(f"it {it} non-finite loss/grad, step skipped ({skipped} total)")
        it += 1
        if it % args.log_every == 0:
            log(f"it {it} k {k} lr {lr:.2e} loss {loss.item():.5f} pos {lp.item():.6f} vel {lv.item():.5f} pen {pen[0].item():.2e}/{pen[1].item():.2e}/{pen[2].item():.2e}/{pen[3].item():.2e} t {time.time() - t_start:.0f}s")
            hist.append((it, k, loss.item()))
        if it % args.val_every == 0 or it == args.iters:
            with torch.no_grad():
                vs = [run_batch(model, pt, Dva, torch.arange(i, min(i + args.batch, Dva["P"].shape[0]), device=dev),
                                torch.zeros(min(args.batch, Dva["P"].shape[0] - i), dtype=torch.long, device=dev), args.k_end) for i in range(0, Dva["P"].shape[0], args.batch)]
            vloss = float(np.mean([100 * a.item() + b.item() for a, b, _ in vs]))
            log(f"it {it} VAL k {args.k_end} loss {vloss:.5f}")
            hist.append((it, "val", vloss))
        if STOP:
            save(base + ".pt")
            log(f"SIGTERM at it {it}: checkpoint saved")
            sys.exit(143)
        if it % args.ckpt_every == 0 and all(torch.isfinite(p).all() for p in model.parameters()):
            save(base + ".pt")
    save(base + ".pt")
    save_atomic({"model": model.state_dict(), "it": it, "args": vars(args)}, base + "_final.pt")
    json.dump({"hist": hist, "skipped": skipped, "seconds": time.time() - t_start, "args": vars(args)}, open(base + ".json", "w"))
    open(base + ".done", "w").write("ok\n")
    log(f"done {it} iters in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
