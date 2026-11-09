"""Interior force offset, corrected quantity (audit row 10, docs/debugging/experiment-log.md 2026-11-08 interior-force-offset-corrected-quantity).
Measures the quantity behind the ~1.55x figure of bh_edge_force_check.py: model |a| on the 128x128 grid (scatter_bh_1b grid_dump, rho scattered from the bodies, Exp E weights,
pair term removed, kpad ext, fp32 state) over the exact fp64 softened point-sum |a| at the 128x128 cell centres (eps 0.5, G 1, all bodies), ratio of means by edge-distance band,
on the black-hole scene (Gaussian sigma 8, clipped +-29, vfac 0.3, c 10). Arms: n and per-body mass. Also splits the model grid acceleration into the kernel term alone and the U-Net term.
Usage: PYTHONPATH=. python scripts/interior_force_offset_corrected_quantity.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/interior_force_offset_corrected_quantity.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts import scatter_bh_1b as sb1
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf

G, H, HALF, EPS = 128, 0.5, 32.0, 0.5
BANDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 8), (8, 16), (16, 33), (8, 33)]
ARMS = (("one-million-unit-mass", 1000000, 1.0, (0, 11, 40)), ("one-hundred-thousand-unit-mass", 100000, 1.0, (0,)),
        ("one-hundred-thousand-mass-ten-same-total-mass", 100000, 10.0, (0,)))


@torch.no_grad()
def exact_at_cells(pos, mass, cx, chunk_elems=3.0e8):
    p = pos.double()
    m = mass.double()
    out = torch.zeros(cx.shape[0], 2, dtype=torch.float64, device=pos.device)
    c = max(1, int(chunk_elems / (2 * p.shape[0])))
    cxd = cx.double()
    for s in range(0, cx.shape[0], c):
        d = p[None] - cxd[s:s + c, None]
        w = ((d ** 2).sum(-1) + EPS ** 2) ** -1.5 * m[None]
        out[s:s + c] = (d * w[..., None]).sum(1)
    return out.norm(dim=-1).float()


@torch.no_grad()
def grid_accs(sim, pos, mom, mass, args):
    m = sim.model
    n = pos.shape[0]
    v = sb.speed(mom.float() * args.mom_scale, args.c)
    x = m.scatter(pos.float()[None], v[None], torch.full((1, n), mass, device=pos.device), torch.ones(1, n, device=pos.device))
    saved = sim.field.clone()
    ak = m.kernel_acc(x[:, :1])
    sim.field = saved.clone()
    acc, _ = sim.grid_force(x)
    sim.field = saved
    return acc[0].norm(dim=0).reshape(-1), ak[0].norm(dim=0).reshape(-1)


def band_rows(mo, ak, ex, edge):
    rows = {}
    for lo, hi in BANDS:
        k = (edge >= lo) & (edge < hi)
        rows[f"{lo}-{hi}"] = dict(cells=int(k.sum()), ratio_of_means=float(mo[k].mean() / ex[k].mean()), kernel_only_ratio_of_means=float(ak[k].mean() / ex[k].mean()),
                                  median_ratio=float(np.median(mo[k] / ex[k])), model_mean=float(mo[k].mean()), exact_mean=float(ex[k].mean()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-divisor", type=int, default=1, help="smoke tests only: divide every arm n")
    a = ap.parse_args()
    dev = torch.device(a.device)
    args = argparse.Namespace(sigma=8.0, vfac=0.3, c=10.0, dt=0.1, seed=a.seed, ic="gaussian", ic_resample=False, mom_scale=1.0)
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", args.dt).to(dev)
    if a.ckpt != "random":
        model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    ops.no_pp(model)
    model.kpad = "ext"
    c = (torch.arange(G, device=dev, dtype=torch.float32) + 0.5) * H - HALF
    cx = torch.stack([c[None, :].expand(G, G), c[:, None].expand(G, G)], -1).reshape(-1, 2)
    edge = (HALF - cx.abs().max(-1).values).cpu().numpy()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name, n, mass, ticks in ARMS:
        n //= a.n_divisor
        if name in res:
            continue
        t0 = time.time()
        pos, mom = sb1.make_ic(n, n, args, torch.float32, torch.float32, 1.0, dev)
        sim = sb1.Sim(model, n, n, args.c, args.dt, torch.float32, torch.float32, 1.0, dev)
        sim.init_kick(pos, mom)
        arm = {"n": n, "mass": mass, "ticks": {}}
        for t in range(max(ticks) + 1):
            if t in ticks:
                mo, ak = grid_accs(sim, pos, mom, mass, args)
                ex = exact_at_cells(pos, torch.full((n,), mass, device=dev), cx).cpu().numpy()
                arm["ticks"][t] = band_rows(mo.cpu().numpy(), ak.cpu().numpy(), ex, edge)
                r = arm["ticks"][t]["8-33"]
                print(f"[{name}] tick {t} interior (edge>=8) ratio {r['ratio_of_means']:.4f} kernel-only {r['kernel_only_ratio_of_means']:.4f} median {r['median_ratio']:.4f}", flush=True)
            if t < max(ticks):
                sim.tick(pos, mom)
        arm["seconds"] = time.time() - t0
        res[name] = arm
        json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
