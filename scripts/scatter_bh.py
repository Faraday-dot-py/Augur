"""Black-hole collapse on the learned scatter-field model, with a ground-truth reference.

One Gaussian cluster of N unit-mass bodies (sigma --sigma, cold: per-component speed --vfac * sqrt(N / (4 sigma))), softened 2D gravity (eps 0.5, dt 0.1),
relativistic kinetics with speed limit --c: the state is momentum p (dp/dt = F), v = p / sqrt(1 + p^2/c^2), so |v| < c; horizon R_s = 2N/c^2. The force is the
learned ScatterField.force (potential grid net + FFT kernel + pair term, fed v(p)), integrated with kick-drift-kick on p. The reference is the exact softened
all-pairs sim (gravity_sim.rollout_torch relativistic, 4 substeps per tick) from the same initial state. Writes results/<tag>_snaps.npz (pos, truth_pos, steps) and
results/<tag>.json (per-tick error, core radius, speed, energy).

Usage: PYTHONPATH=. python scripts/scatter_bh.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 300 --steps 100 --tag scatter_bh_300
"""
import argparse
import json

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf


def speed(p, c):
    return p / (1 + (p ** 2).sum(-1, keepdim=True) / c ** 2).sqrt()


def core_radius(pos):
    c = np.median(pos, axis=0)
    return float(np.percentile(np.linalg.norm(pos - c, axis=1), 45))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--vfac", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--tag", default="scatter_bh")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    n = args.n
    pos0 = np.clip(rng.normal(0.0, args.sigma, (n, 2)), -29.0, 29.0)
    vel0 = rng.normal(0.0, args.vfac * np.sqrt(n / (4 * args.sigma)), (n, 2))
    vel0 -= vel0.mean(0)
    c = args.c

    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", args.dt).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()

    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(vel0, dtype=torch.float32, device=dev)[None]
    mass = torch.ones(1, n, device=dev)
    mask = torch.ones(1, n, device=dev)
    vmag = vel.norm(dim=-1, keepdim=True).clamp(max=0.99 * c)
    mom = vel * (1 - (vmag / c) ** 2).clamp(min=1e-6).rsqrt()
    field = model.init_field(1, dev)[0]
    dt = args.dt
    snaps = [pos[0].cpu().numpy()]
    vmax_c = []
    out_of_grid = []
    with torch.no_grad():
        a, field = model.force(pos, speed(mom, c), mass, mask, field)
        for _ in range(args.steps):
            mom = mom + 0.5 * dt * a
            pos = pos + dt * speed(mom, c)
            a, field = model.force(pos, speed(mom, c), mass, mask, field)
            mom = mom + 0.5 * dt * a
            snaps.append(pos[0].cpu().numpy())
            vmax_c.append(float(speed(mom, c).norm(dim=-1).max() / c))
            out_of_grid.append(float((pos.abs() > model.extent / 2).any(-1).float().mean()))
    snaps = np.stack(snaps)

    Pt, Vt = gs.rollout_torch(pos0, vel0, args.steps, dev, dt=dt, substeps=4, eps=args.eps, relativistic=True, c=c)
    Pt = np.asarray(Pt)
    err = np.linalg.norm(snaps - Pt, axis=-1).mean(1)
    rm = [core_radius(f) for f in snaps]
    rt = [core_radius(f) for f in Pt]
    res = {"n": n, "sigma": args.sigma, "vfac": args.vfac, "c": c, "R_s": 2 * n / c ** 2, "steps": args.steps, "ckpt": args.ckpt,
           "err": err.tolist(), "core_radius_model": rm, "core_radius_truth": rt, "vmax_over_c_model": vmax_c,
           "frac_outside_grid": out_of_grid}
    print(f"err@5/10/20/50/100 {[round(float(err[i]), 3) for i in (5, 10, 20, 50, 100) if i < len(err)]}")
    print(f"core radius model/truth at 0,10,20,50,100: {[(round(rm[i], 2), round(rt[i], 2)) for i in (0, 10, 20, 50, 100) if i < len(rm)]}")
    print(f"vmax/c model max {max(vmax_c):.3f}, max frac outside grid {max(out_of_grid):.3f}")
    np.savez(f"results/{args.tag}_snaps.npz", pos=snaps, truth_pos=Pt, steps=np.arange(args.steps + 1))
    json.dump(res, open(f"results/{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
