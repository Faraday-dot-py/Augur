"""scatter_bh.py at large N: chunked exact-kNN pair term for the model, chunked exact all-pairs reference (scatter_scaling_ops), model snaps saved before the reference runs.

Usage: PYTHONPATH=. python scripts/scatter_bh_large.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 100000 --steps 100 --tag scatter_bh_100000
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts import scatter_scaling_ops as ops
from scripts import train_scatter_field as tsf


@torch.no_grad()
def exact_rollout_rel(pos, vel, steps, dt, substeps, eps, c):
    h = dt / substeps
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
    ps = [pos.cpu().numpy()]
    a = ops.exact_acc(pos, eps)
    for _ in range(steps):
        for _ in range(substeps):
            mom = mom + 0.5 * h * a
            pos = pos + h * sb.speed(mom, c)
            a = ops.exact_acc(pos, eps)
            mom = mom + 0.5 * h * a
        ps.append(pos.cpu().numpy())
    return np.stack(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--tag", default="scatter_bh_large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kpad", choices=["zero", "ext"], default="zero")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    n, c, dt = args.n, args.c, args.dt
    pos0 = np.clip(rng.normal(0.0, args.sigma, (n, 2)), -29.0, 29.0)
    vel0 = rng.normal(0.0, args.vfac * np.sqrt(n / (4 * args.sigma)), (n, 2))
    vel0 -= vel0.mean(0)

    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", dt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    model.kpad = args.kpad
    ops.chunked_pp(model)

    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
    field = model.init_field(1, dev)[0]
    snaps = [pos[0].cpu().numpy()]
    vmax_c, out_of_grid = [], []
    with torch.no_grad():
        a, field = model.force(pos, sb.speed(mom, c), mass, mask, field)
        for t in range(args.steps):
            mom = mom + 0.5 * dt * a
            pos = pos + dt * sb.speed(mom, c)
            a, field = model.force(pos, sb.speed(mom, c), mass, mask, field)
            mom = mom + 0.5 * dt * a
            snaps.append(pos[0].cpu().numpy())
            vmax_c.append(float(sb.speed(mom, c).norm(dim=-1).max() / c))
            out_of_grid.append(float((pos.abs() > model.extent / 2).any(-1).float().mean()))
            if t % 10 == 0:
                print(f"model tick {t + 1} frac outside grid {out_of_grid[-1]:.3f}", flush=True)
    snaps = np.stack(snaps)
    np.savez(f"results/{args.tag}_snaps.npz", pos=snaps, steps=np.arange(args.steps + 1))
    del model, field, a, mom, pos, vel, mass, mask
    torch.cuda.empty_cache()

    Pt = exact_rollout_rel(torch.tensor(pos0, dtype=torch.float64, device=dev), torch.tensor(vel0, dtype=torch.float64, device=dev),
                           args.steps, dt, 4, args.eps, c)
    err = np.linalg.norm(snaps - Pt, axis=-1).mean(1)
    cv = np.linalg.norm(pos0[None] + (np.arange(args.steps + 1)[:, None, None] * dt) * np.minimum(1.0, 0.99 * c / np.maximum(np.linalg.norm(vel0, axis=1, keepdims=True), 1e-9)) * vel0[None] - Pt, axis=-1).mean(1)
    rm = [sb.core_radius(f) for f in snaps]
    rt = [sb.core_radius(f) for f in Pt]
    res = {"n": n, "sigma": args.sigma, "vfac": args.vfac, "c": c, "R_s": 2 * n / c ** 2, "steps": args.steps, "ckpt": args.ckpt,
           "err": err.tolist(), "constvel_err": cv.tolist(), "core_radius_model": rm, "core_radius_truth": rt,
           "vmax_over_c_model": vmax_c, "frac_outside_grid": out_of_grid}
    print(f"err@5/10/20/50/100 {[round(float(err[i]), 3) for i in (5, 10, 20, 50, 100) if i < len(err)]}")
    print(f"constvel err@5/10/20 {[round(float(cv[i]), 3) for i in (5, 10, 20) if i < len(cv)]}")
    print(f"core radius model/truth at 0,10,20,50,100: {[(round(rm[i], 2), round(rt[i], 2)) for i in (0, 10, 20, 50, 100) if i < len(rm)]}")
    print(f"vmax/c model max {max(vmax_c):.3f}, max frac outside grid {max(out_of_grid):.3f}")
    np.savez(f"results/{args.tag}_snaps.npz", pos=snaps, truth_pos=Pt, steps=np.arange(args.steps + 1))
    json.dump(res, open(f"results/{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
