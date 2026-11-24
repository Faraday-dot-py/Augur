"""Bit-output cell model at a 64x64 grid with a long-range law, against its float twin (experiment bit-output-cell-model-at-64-grid-with-long-range-law).
Candidate = plain 8-bit two's-complement bits per axis (BCE) decoded by soft decode; twin = float regression, same batch stream and first-layer init.
Pre-registration: docs/debugging/architecture-push-prereg/bit-output-cell-model-at-64-grid-with-long-range-law.md.
Toy sanity path (reproduces the 2026-11-08 centered setup, oracle force error 0.596): --grid 8 --n 48 --law short --coarse "" --cal-scenes 2048 --eval-scenes 512 --rollout-scenes 256
Usage: PYTHONPATH=scripts python scripts/bit_output_64_grid_long_range.py --out-dir results/bit_output_64_grid [--resume] [--fresh]
"""
import argparse
import json
import os
import signal
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bit_force_smoke import pair_force as short_force
from bit_output_encoding_fixes import Codec

DT = 0.1
NBITS = 8
VAL_SEED = 9000
HELD_SEEDS = (9100, 9200, 9300)
FRESH_SEEDS = (12000, 12100, 12200)
RATIO_PASS, RATIO_KILL = 1.5, 3.0
PLATEAU_DECLINE = 0.03
STOP = {"flag": False}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


class Geo:
    def __init__(self, grid, n, law, coarse, window, k, eps):
        self.g, self.n, self.law, self.coarse, self.w, self.k, self.eps = grid, n, law, coarse, window, k, eps
        for s in coarse:
            assert grid % s == 0 and grid // s >= window, f"coarse factor {s} invalid for grid {grid}"
        self.din = 3 * window * window * (1 + len(coarse))

    def force(self, pos, chunk_pairs=2 ** 27):
        if self.law == "short":
            assert self.g == 8, "short law is the 8x8 toy only"
            return short_force(pos)
        B, N = pos.shape[:2]
        step = max(1, chunk_pairs // (N * N))
        out = []
        for b0 in range(0, B, step):
            p = pos[b0:b0 + step]
            d = p[:, :, None, :] - p[:, None, :, :]
            q = (d ** 2).sum(-1) + self.eps ** 2
            w = q ** -1.5
            w = w * (1 - torch.eye(N, device=pos.device))[None]
            out.append(-self.k * (w[..., None] * d).sum(2))
        return torch.cat(out)

    def cell_ids(self, pos, s=1):
        gs = self.g // s
        c = (pos / s).floor().clamp(0, gs - 1).long()
        return c[..., 0] * gs + c[..., 1], c

    def cell_mean(self, pos, val):
        B = pos.shape[0]
        idx, _ = self.cell_ids(pos)
        gg = self.g * self.g
        s = torch.zeros(B, gg, val.shape[-1], device=pos.device).scatter_add_(1, idx[..., None].expand(-1, -1, val.shape[-1]), val)
        n = torch.zeros(B, gg, device=pos.device).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float))
        return s / n.clamp_min(1)[..., None], n

    def windows(self, grid):
        W = self.w
        p = F.pad(grid, (W // 2, W // 2, W // 2, W // 2))
        w = p.unfold(2, W, 1).unfold(3, W, 1)[:, :, :grid.shape[2], :grid.shape[3]]
        return w.permute(0, 2, 3, 1, 4, 5).reshape(grid.shape[0], grid.shape[2], grid.shape[3], -1)

    def features(self, pos):
        B, g = pos.shape[0], self.g
        parts = []
        n_fine = None
        for s in (1,) + tuple(self.coarse):
            gs = g // s
            idx, c = self.cell_ids(pos, s)
            off = pos / s - c.float() - 0.5
            n = torch.zeros(B, gs * gs, device=pos.device).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float))
            mo = torch.zeros(B, gs * gs, 2, device=pos.device).scatter_add_(1, idx[..., None].expand(-1, -1, 2), off) / n.clamp_min(1)[..., None]
            dens = n / (s * s) if s > 1 else n
            grid = torch.cat([dens[..., None], mo], -1).reshape(B, gs, gs, 3).permute(0, 3, 1, 2)
            f = self.windows(grid)
            if s > 1:
                f = f[:, :, None, :, None].expand(B, gs, s, gs, s, f.shape[-1])
            parts.append(f.reshape(B, g * g, -1))
            if s == 1:
                n_fine = n
        return torch.cat(parts, -1), n_fine

    def apply(self, pos, cell_dp):
        idx, _ = self.cell_ids(pos)
        return torch.gather(cell_dp, 1, idx[..., None].expand(-1, -1, 2)) / DT

    def sample(self, B, gen, device):
        return torch.rand(B, self.n, 2, generator=gen, device=device) * self.g

    def rollout(self, pos0, vel0, fn, steps):
        pos, vel = pos0.clone(), vel0.clone()
        traj = []
        for _ in range(steps):
            vel = vel + fn(pos) * DT
            pos = pos + vel * DT
            low, high = pos < 0, pos > self.g
            pos = torch.where(low, -pos, pos)
            pos = torch.where(high, 2 * self.g - pos, pos)
            vel = torch.where(low | high, -vel, vel)
            traj.append(pos.clone())
        return traj


def make_net(din, out, hidden=128):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


def calibrate(geo, seed, scenes, dev, chunk=16):
    gen = torch.Generator(device=dev).manual_seed(seed)
    vals = []
    for b0 in range(0, scenes, chunk):
        pos = geo.sample(min(chunk, scenes - b0), gen, dev)
        dp, _ = geo.cell_mean(pos, geo.force(pos) * DT)
        vals.append(dp.abs().flatten().cpu())
    allv = torch.cat(vals)
    if allv.numel() > 2 ** 24:
        allv = allv[torch.randperm(allv.numel(), generator=torch.Generator().manual_seed(seed))[:2 ** 24]]
    return (allv.quantile(0.999) / 127).item()


def make_val(geo, scenes, dev):
    gen = torch.Generator(device=dev).manual_seed(VAL_SEED)
    pos = geo.sample(scenes, gen, dev)
    feats, n = geo.features(pos)
    tgt, _ = geo.cell_mean(pos, geo.force(pos) * DT)
    return feats, tgt, n > 0


def loss_fn(net, kind, codec, scale, feats, tgt, mask):
    mask = mask.float()
    y = net(feats)
    if kind == "float":
        return (((y - tgt / scale) ** 2).mean(-1) * mask).sum() / mask.sum()
    t = codec.encode(tgt).reshape(*tgt.shape[:2], -1)
    return (F.binary_cross_entropy_with_logits(y, t, reduction="none").mean(-1) * mask).sum() / mask.sum()


def save_atomic(path, obj):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def train_cell(args, geo, seed, kind, codec, scale, val, dev):
    path = f"{args.out_dir}/ckpt/{seed}_{kind}.pt"
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed + 100)
    net = make_net(geo.din, 2 * NBITS if kind == "bits" else 2).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    step, vals = 0, []
    if os.path.exists(path):
        if not args.resume:
            raise SystemExit(f"{path} exists; pass --resume to continue it")
        st = torch.load(path, map_location=dev, weights_only=False)
        assert st["steps"] == args.steps and st["scale"] == scale, "checkpoint does not match this run"
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"].cpu())
        step, vals = st["step"], st["val"]
        log(f"{kind} seed {seed} resumed at step {step}/{args.steps}")

    def save():
        save_atomic(path, {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "gen": gen.get_state(), "val": vals, "scale": scale,
                           "steps": args.steps})

    t0 = time.time()
    while step < args.steps:
        if STOP["flag"]:
            save()
            log("SIGTERM: checkpoint saved")
            sys.exit(143)
        pos = geo.sample(args.bs, gen, dev)
        feats, n = geo.features(pos)
        tgt, _ = geo.cell_mean(pos, geo.force(pos) * DT)
        loss = loss_fn(net, kind, codec, scale, feats, tgt, n > 0)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % args.val_every == 0 or step == args.steps:
            with torch.no_grad():
                vals.append((step, float(loss_fn(net, kind, codec, scale, *val))))
        if step % args.log_every == 0:
            log(f"{kind} seed {seed} step {step} loss {loss.item():.4f} val {vals[-1][1]:.4f} t {time.time() - t0:.0f}s")
        if step % args.ckpt_every == 0 or step == args.steps:
            save()
    return net, vals


def plateau(vals, steps):
    last = [v for s, v in vals if s > 0.9 * steps]
    prev = [v for s, v in vals if 0.8 * steps < s <= 0.9 * steps]
    if not last or not prev:
        raise SystemExit("plateau guard: no validation points in the last or preceding 10% window (check --val-every)")
    a, b = sum(prev) / len(prev), sum(last) / len(last)
    decline = (a - b) / a
    return {"mean_preceding": a, "mean_last": b, "decline": decline, "undertrained": bool(decline > PLATEAU_DECLINE)}


def make_eval(geo, seed, args, dev):
    gen = torch.Generator(device=dev).manual_seed(seed)
    pos = geo.sample(args.eval_scenes, gen, dev)
    e = {"pos": pos, "F": geo.force(pos)}
    e["vel0"] = torch.randn(args.rollout_scenes, geo.n, 2, generator=gen, device=dev) * 0.3
    e["pos0"] = geo.sample(args.rollout_scenes, gen, dev)
    e["true"] = geo.rollout(e["pos0"], e["vel0"], geo.force, 20)
    return e


def predict(geo, net, kind, codec, scale, pos, soft, chunk=32):
    out = []
    for b0 in range(0, pos.shape[0], chunk):
        feats, _ = geo.features(pos[b0:b0 + chunk])
        y = net(feats)
        if kind == "float":
            out.append(y * scale)
        else:
            yb = y.reshape(*y.shape[:2], 2, NBITS)
            out.append(codec.soft_decode(yb) if soft else codec.decode((yb > 0).float()))
    return torch.cat(out)


def rel(p, Ft):
    return ((p - Ft).norm(dim=-1).sum() / Ft.norm(dim=-1).sum()).item()


def rollout_err(geo, e, fn):
    with torch.no_grad():
        tr = geo.rollout(e["pos0"], e["vel0"], fn, 20)
    return {f"err@{t}": (tr[t - 1] - e["true"][t - 1]).norm(dim=-1).mean().item() for t in (5, 10, 20)}


def references(geo, e):
    def oracle(p):
        return geo.apply(p, geo.cell_mean(p, geo.force(p) * DT)[0])
    return {"zero_force_F": 1.0, "oracle_cell_mean_F": rel(oracle(e["pos"]), e["F"]), "zero_force_R": rollout_err(geo, e, lambda p: torch.zeros_like(p)),
            "oracle_cell_mean_R": rollout_err(geo, e, oracle)}


def evaluate(geo, net, kind, codec, scale, e, soft):
    with torch.no_grad():
        F_ = rel(geo.apply(e["pos"], predict(geo, net, kind, codec, scale, e["pos"], soft)), e["F"])
        fn = lambda p: geo.apply(p, predict(geo, net, kind, codec, scale, p, soft, chunk=64))
        return {"F": F_, "R": rollout_err(geo, e, fn)}


def score(res, seed, eval_seeds, steps):
    runs = {s: res["runs"][f"{seed}/{s}"] for s in eval_seeds}
    mean = lambda arm, get: sum(get(runs[s][arm]) for s in eval_seeds) / len(eval_seeds)
    out = {"train_seed": seed, "eval_seeds": list(eval_seeds), "candidate": "plain-bits-soft-decode", "twin_per_eval_seed": {}}
    for arm in ("bits_soft", "bits_hard"):
        out[arm] = {"F_ratio": mean(arm, lambda r: r["F"]) / mean("float", lambda r: r["F"]),
                    "R20_ratio": mean(arm, lambda r: r["R"]["err@20"]) / mean("float", lambda r: r["R"]["err@20"]),
                    "per_eval_seed": {str(s): {"F_ratio": runs[s][arm]["F"] / runs[s]["float"]["F"], "R20_ratio": runs[s][arm]["R"]["err@20"] / runs[s]["float"]["R"]["err@20"]}
                                      for s in eval_seeds}}
    twinF = [runs[s]["float"]["F"] for s in eval_seeds]
    twinR = [runs[s]["float"]["R"]["err@20"] for s in eval_seeds]
    sd = lambda x: (sum((v - sum(x) / len(x)) ** 2 for v in x) / len(x)) ** 0.5 / (sum(x) / len(x))
    out["twin_noise_floor"] = {"F_per_eval_seed": twinF, "R20_per_eval_seed": twinR, "F_rel_spread": sd(twinF), "R20_rel_spread": sd(twinR)}
    fr, rr = out["bits_soft"]["F_ratio"], out["bits_soft"]["R20_ratio"]
    label = "PASS" if fr <= RATIO_PASS and rr <= RATIO_PASS else "KILL" if fr > RATIO_KILL or rr > RATIO_KILL else "INCONCLUSIVE"
    pl = {k: plateau(res["val"][f"{seed}/{k}"], steps) for k in ("float", "bits")}
    out["plateau"] = pl
    if label != "PASS" and any(p["undertrained"] for p in pl.values()):
        label = "INCONCLUSIVE-undertrained"
    out["label"] = label
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def dump(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--law", choices=["long", "short"], default="long")
    ap.add_argument("--coarse", default="2,4,8")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seeds", default="4738")
    ap.add_argument("--eval-seeds", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--cal-scenes", type=int, default=128)
    ap.add_argument("--eval-scenes", type=int, default=128)
    ap.add_argument("--rollout-scenes", type=int, default=64)
    ap.add_argument("--val-scenes", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_default_device(dev)
    os.makedirs(f"{a.out_dir}/ckpt", exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__("flag", True))
    seeds = [int(s) for s in a.seeds.split(",")]
    eval_seeds = [int(s) for s in a.eval_seeds.split(",")] if a.eval_seeds else list(FRESH_SEEDS if a.fresh else HELD_SEEDS)
    tag = "held-out" if eval_seeds == list(HELD_SEEDS) else "fresh" if eval_seeds == list(FRESH_SEEDS) else "custom"
    coarse = tuple(int(s) for s in a.coarse.split(",") if s)
    geo = Geo(a.grid, a.n, a.law, coarse, a.window, a.k, a.eps)
    rp = f"{a.out_dir}/results.json"
    if os.path.exists(rp) and not a.resume:
        raise SystemExit(f"{rp} exists; pass --resume to continue it")
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    res.setdefault("meta", {k: v for k, v in vars(a).items()})
    for k in ("runs", "val", "scale", "refs"):
        res.setdefault(k, {})
    val = make_val(geo, a.val_scenes, dev)
    evals = {s: make_eval(geo, s, a, dev) for s in eval_seeds}
    for s in eval_seeds:
        res["refs"].setdefault(f"{tag}/{s}", references(geo, evals[s]))
    dump(rp, res)
    log(f"grid {a.grid} n {a.n} law {a.law} coarse {coarse} din {geo.din} steps {a.steps} seeds {seeds} eval {tag} {eval_seeds}")
    for seed in seeds:
        scale = res["scale"].get(str(seed)) or calibrate(geo, seed, a.cal_scenes, dev)
        res["scale"][str(seed)] = scale
        codec = Codec(NBITS, scale)
        nets = {}
        for kind in ("float", "bits"):
            nets[kind], vals = train_cell(a, geo, seed, kind, codec, scale, val, dev)
            res["val"][f"{seed}/{kind}"] = vals
            dump(rp, res)
        for s in eval_seeds:
            key = f"{seed}/{s}"
            if key in res["runs"]:
                continue
            res["runs"][key] = {"float": evaluate(geo, nets["float"], "float", None, scale, evals[s], False),
                                "bits_soft": evaluate(geo, nets["bits"], "bits", codec, scale, evals[s], True),
                                "bits_hard": evaluate(geo, nets["bits"], "bits", codec, scale, evals[s], False)}
            dump(rp, res)
        res.setdefault("verdicts", {})[f"{seed}/{tag}"] = score(res, seed, eval_seeds, a.steps)
        dump(rp, res)
        log(f"verdict seed {seed} {tag}: {json.dumps(jsonable(res['verdicts'][f'{seed}/{tag}']['label']))} soft F {res['verdicts'][f'{seed}/{tag}']['bits_soft']['F_ratio']:.3f} "
            f"R20 {res['verdicts'][f'{seed}/{tag}']['bits_soft']['R20_ratio']:.3f}")
    assert os.path.getsize(rp) > 0
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
