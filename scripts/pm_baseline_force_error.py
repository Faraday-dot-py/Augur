"""Static force error of the learned scatter-field model versus a plain same-grid particle-mesh (PM) comparator.
Docs: docs/debugging/top8-first-pass-2026-10-29.md row 2; pre-registration in docs/debugging/supervisor-log.md (2026-11-03).

Learned model = pinned gravity model kern/E_frozen.pt (ms_kp_pot_v_g128, kernel + U-Net + pair term, momentum fix). PM comparator = same G=128 grid, same CIC deposit and gather,
same finite-difference gradient, FFT convolution with the analytic softened-gravity kernel -1/sqrt(r^2+eps^2); no U-Net, no pair term.
Truth = fp64 softened all-pairs sum (scatter_scaling_ops.exact_acc). Error per particle = |a - a_true| / rms(|a_true|) of its scene.
Usage: PYTHONPATH=. python scripts/pm_baseline_force_error.py --ckpt kern/E_frozen.pt --out results/pm_baseline_force_error.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_scaling_ops as ops

EPS = 0.5
LIM = 28.0
FAMILIES = ("plummer", "two_clump", "exp_disk")
NS = {300: 16, 500: 16, 1000: 16, 2000: 8, 4000: 4, 10000: 3, 100000: 1}


def _inside(p):
    return (np.abs(p) < LIM).all(1)


def sample(family, n, rng, scale=1.0):
    out = np.zeros((0, 2))
    while len(out) < n:
        m = 2 * n
        if family == "plummer":
            u = rng.uniform(0.0, 0.97, m)
            R = 4.0 * np.sqrt(u / (1 - u))
            th = rng.uniform(0, 2 * np.pi, m)
            p = np.stack([R * np.cos(th), R * np.sin(th)], 1)
        elif family == "uniform_disk":
            R = 24.0 * np.sqrt(rng.uniform(0, 1, m))
            th = rng.uniform(0, 2 * np.pi, m)
            p = np.stack([R * np.cos(th), R * np.sin(th)], 1)
        elif family == "hernquist":
            u = rng.uniform(0.0, 0.97, m)
            R = 6.0 * np.sqrt(u) / (1 - np.sqrt(u))
            th = rng.uniform(0, 2 * np.pi, m)
            p = np.stack([R * np.cos(th), R * np.sin(th)], 1)
        elif family == "three_clump":
            k = rng.integers(0, 3, m)
            cen = np.array([[-12.0, -8.0], [10.0, -6.0], [0.0, 13.0]])[k]
            p = rng.normal(0, 3.5, (m, 2)) + cen
        elif family == "exp_disk":
            R = rng.gamma(2.0, 5.0, m)
            th = rng.uniform(0, 2 * np.pi, m)
            p = np.stack([R * np.cos(th), R * np.sin(th)], 1)
        else:
            big = rng.random(m) < 2 / 3
            p = np.where(big[:, None], rng.normal(0, 4.0, (m, 2)) + [-9.0, 0.0], rng.normal(0, 3.0, (m, 2)) + [11.0, 0.0])
        p = p * scale
        out = np.concatenate([out, p[_inside(p)]])
    out = out[:n]
    if family in ("two_clump", "three_clump"):
        a = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(a), np.sin(a)
        out = out @ np.array([[c, -s], [s, c]])
        out = np.clip(out, -LIM, LIM)
    return out


def pm_acc(model, pos, vel, mass, mask):
    G, h = model.grid, model.h
    rho = model.scatter(pos, vel, mass, mask)[:, :1]
    idx = torch.arange(2 * G, device=pos.device, dtype=pos.dtype)
    d = torch.minimum(idx, 2 * G - idx) * h
    K = -1.0 / torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2 + EPS ** 2)
    full = torch.fft.irfft2(torch.fft.rfft2(rho, s=(2 * G, 2 * G)) * torch.fft.rfft2(K), s=(2 * G, 2 * G)) * h ** 2
    acc = model.neg_grad(full[..., :G, :G])
    return model.gather(acc, pos) * mask[..., None]


@torch.no_grad()
def learned_acc(model, pos, vel, mass, mask, warm):
    field = model.init_field(pos.shape[0], pos.device)[0]
    a, field = model.force(pos, vel, mass, mask, field)
    for _ in range(warm):
        a, field = model.force(pos, vel, mass, mask, field)
    return a


def errs(a, at, mask):
    rms = ((at.norm(dim=-1) ** 2 * mask).sum(1) / mask.sum(1)).sqrt()[:, None]
    e = ((a - at).norm(dim=-1) / rms)[mask > 0]
    rel = ((a - at).norm(dim=-1) / at.norm(dim=-1).clamp_min(1e-12))[mask > 0]
    return e, rel


def stats(x):
    return {"median": float(x.median()), "p90": float(x.quantile(0.9)), "mean": float(x.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--ns", default="300,1000,10000,100000")
    ap.add_argument("--scene-mult", type=int, default=1)
    ap.add_argument("--density-scale", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = sf.ScatterField(grid=128, extent=64.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, in_scale=1.0, potential=True,
                            kernel=True, pp=2.0, split=True, verlet=True).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    rng = np.random.default_rng(a.seed)
    res = {"seed": a.seed, "warm": a.warm, "eps": EPS, "grid": 128, "extent": 64.0}
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            ns = NS[n] * (a.scene_mult if n <= 1000 else 1)
            pos = torch.tensor(np.stack([sample(fam, n, rng, np.sqrt(n / 1000) if a.density_scale and n > 1000 else 1.0) for _ in range(ns)]), dtype=torch.float32, device=dev)
            if n >= 10000:
                model.opt.add("adapt")
            vel = torch.zeros_like(pos)
            mass = torch.ones(ns, n, device=dev)
            mask = torch.ones(ns, n, device=dev)
            at = torch.stack([ops.exact_acc(p.double(), EPS) for p in pos]).float()
            with torch.no_grad():
                accs = {"learned_warm": learned_acc(model, pos, vel, mass, mask, a.warm), "learned_zero_field": learned_acc(model, pos, vel, mass, mask, 0),
                        "plain_pm": pm_acc(model, pos, vel, mass, mask)}
            row = {}
            for k, v in accs.items():
                e, rel = errs(v, at, mask)
                row[k] = {"norm_by_rms": stats(e), "rel_per_particle": stats(rel)}
            res[f"{fam}_n{n}"] = row
            print(fam, n, {k: (round(v["norm_by_rms"]["median"], 5), round(v["norm_by_rms"]["p90"], 5)) for k, v in row.items()}, flush=True)
            model.opt.discard("adapt")
            del pos, vel, mass, mask, at, accs
            torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
