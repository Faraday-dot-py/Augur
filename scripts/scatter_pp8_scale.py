"""Scalability of the exact-pair radius pp (pp=8, knn=48, prior kernel) beyond N=100.
Force level: model force at N bodies vs exact softened all-pairs (fp64, chunked), by pair-term implementation (dense / cell / adapt),
rel RMS error overall and split by the number of neighbours within pp (knn truncation: n_pp > knn). Untrained (zero-init, analytic-floor)
models by default, or a trained checkpoint (--ckpt). Rollout level (N <= 3000): err@5/10/20 vs 4-substep exact truth.
Scenes: 'gauss' (sigma 8, clipped +-28.8, the scatter_bh scene) and 'unif' (uniform in +-28.8).
Usage: PYTHONPATH=. python scripts/scatter_pp8_scale.py --variant ms_kp_pot_v_g128_prior_pp8 --out results/scatter_pp8/scale_pair_floor.json"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as ts

EPS, DT = 0.5, 0.1
LIM = 28.8
NS = [100, 300, 1000, 3000, 10000, 30000, 100000]
IMPLS = ("dense", "cell", "adapt")
DENSE_MAX = 3000


def sync():
    torch.cuda.synchronize()


def scene(kind, n, seed):
    rng = np.random.default_rng([seed, n])
    if kind == "gauss":
        pos = np.clip(rng.normal(0.0, 8.0, (n, 2)), -LIM, LIM)
    else:
        pos = rng.uniform(-LIM, LIM, (n, 2))
    vel = rng.normal(0.0, 0.3 * np.sqrt(n / 32.0), (n, 2))
    vel -= vel.mean(0)
    return pos, vel


@torch.no_grad()
def exact_acc(pos, chunk=2048):
    n = pos.shape[0]
    out = torch.zeros_like(pos)
    for i in range(0, n, chunk):
        d = pos[None, :, :] - pos[i:i + chunk, None, :]
        w = ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
        w[torch.arange(min(chunk, n - i)), torch.arange(i, min(i + chunk, n))] = 0.0
        out[i:i + chunk] = (d * w[..., None]).sum(1)
    return out


@torch.no_grad()
def count_within(pos, r, chunk=2048):
    n = pos.shape[0]
    out = torch.zeros(n, dtype=torch.long, device=pos.device)
    for i in range(0, n, chunk):
        d2 = ((pos[None, :, :] - pos[i:i + chunk, None, :]) ** 2).sum(-1)
        out[i:i + chunk] = (d2 < r * r).sum(1) - 1
    return out


def make(variant, ckpt, impl, dev):
    m = ts.build(ts.EXPS["E"], variant, DT).to(dev)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    m.eval()
    if impl != "dense":
        sf.apply_opt(m, impl)
    return m


def rel(a, at, sel):
    return float((((a - at)[sel] ** 2).sum() / (at[sel] ** 2).sum()).sqrt()) if int(sel.sum()) else None


def force_stage(args, dev):
    res = {}
    for kind in ("gauss", "unif"):
        for n in NS:
            pos, vel = scene(kind, n, 4738)
            p64 = torch.tensor(pos, device=dev)
            at = exact_acc(p64).float()
            P = p64.float()[None]
            V = torch.tensor(vel, device=dev, dtype=torch.float32)[None]
            ones = torch.ones(1, n, device=dev)
            pp = ts.VARIANTS[args.variant].get("pp", 2.0)
            knn = ts.VARIANTS[args.variant].get("knn", 16)
            cnt = count_within(P[0], pp)
            trunc = cnt > knn
            rec = {"frac_trunc": float(trunc.float().mean()), "mean_n_pp": float(cnt.float().mean()), "max_n_pp": int(cnt.max())}
            for impl in IMPLS:
                if (impl == "dense" and n > DENSE_MAX) or (impl == "cell" and kind == "gauss" and n > 10000):
                    continue
                m = make(args.variant, args.ckpt, impl, dev)
                f0 = m.init_field(1, dev)[0]
                try:
                    with torch.no_grad():
                        a = m.force(P, V, ones, ones, f0)[0][0]
                        sync()
                        torch.cuda.reset_peak_memory_stats(dev)
                        base = torch.cuda.memory_allocated(dev)
                        ts_ = []
                        for _ in range(5 if n < 10000 else 3):
                            t = time.perf_counter()
                            m.force(P, V, ones, ones, f0)
                            sync()
                            ts_.append((time.perf_counter() - t) * 1000)
                        peak = (torch.cuda.max_memory_allocated(dev) - base) / 2 ** 20
                except torch.OutOfMemoryError:
                    rec[impl] = {"oom": True}
                    torch.cuda.empty_cache()
                    continue
                rec[impl] = {"rel_all": rel(a, at, torch.ones(n, dtype=torch.bool, device=dev)), "rel_untrunc": rel(a, at, ~trunc),
                             "rel_trunc": rel(a, at, trunc), "ms": float(np.median(ts_)), "peak_mb": peak}
                del m, a
            res[f"{kind}_{n}"] = rec
            print(kind, n, json.dumps(rec), flush=True)
            json.dump(res, open(args.out, "w"))
    return res


def rollout_stage(args, dev, res):
    for kind in ("gauss", "unif"):
        for n in (300, 1000, 3000):
            pos, vel = scene(kind, n, 4738)
            Pt, Vt = gs.rollout_torch(pos, vel, 20, dev, dt=DT, substeps=4, eps=EPS)
            Pt = torch.tensor(np.asarray(Pt), device=dev)
            for impl in ("dense", "cell"):
                m = make(args.variant, args.ckpt, impl, dev)
                P0 = torch.tensor(pos, device=dev, dtype=torch.float32)[None]
                V0 = torch.tensor(vel, device=dev, dtype=torch.float32)[None]
                ones = torch.ones(1, n, device=dev)
                Pm, _ = sf.rollout(m, P0, V0, ones, ones, 20)
                err = (Pm[0].double() - Pt).norm(dim=-1).mean(-1)
                res[f"{kind}_{n}"].setdefault(impl, {})["roll_err_5_10_20"] = [float(err[k]) for k in (5, 10, 20)]
                print("rollout", kind, n, impl, [round(float(err[k]), 5) for k in (5, 10, 20)], flush=True)
                del m
    json.dump(res, open(args.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-rollout", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    res = force_stage(args, dev)
    if not args.no_rollout:
        rollout_stage(args, dev, res)


if __name__ == "__main__":
    main()
