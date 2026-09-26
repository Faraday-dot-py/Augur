"""Scale-generalization ("tiling") test for a pair-force gravity model trained on a small
region: same constant density, larger area and body count, error vs the exact analytic sim.
Variants: model cutoff radius, all pairs with raw f(d) extrapolation, all pairs with log d
clamped to the trained range (1/d^2 tail). Also checks strip tiling with halo == cutoff
against the global model force."""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from model.central_force import CentralForceDynamics
from scripts import gravity_sim as gs


class ClampLog(nn.Module):
    def __init__(self, hi):
        super().__init__()
        self.hi = hi

    def forward(self, x):
        return x.clamp(max=self.hi)


def energy(pos, vel, eps, chunk=2048):
    pe = 0.0
    for i in range(0, len(pos), chunk):
        d = pos[None, :, :] - pos[i:i + chunk, None, :]
        r = torch.sqrt((d ** 2).sum(-1) + eps ** 2)
        inv = 1.0 / r
        idx = torch.arange(i, min(i + chunk, len(pos)), device=pos.device)
        inv[torch.arange(len(idx)), idx] = 0.0
        pe = pe + inv.sum()
    return (0.5 * (vel ** 2).sum() - 0.5 * pe).item()


def trained_dmax(rng_n, steps, seed, eps, dt, device):
    data = gs.make_dataset(8, rng_n, steps, seed, scale=True, device=device, dt=dt, eps=eps)
    m = 0.0
    for P, _ in data:
        for t in (0, len(P) // 2, len(P) - 1):
            p = torch.tensor(P[t], device=device)
            m = max(m, torch.cdist(p, p).max().item())
    return m


def rollout_model(dyn, p0, v0, k, dt):
    pos, vel = p0, v0
    hidden = torch.zeros(pos.shape[0], dyn.hidden_dim, device=pos.device)
    ps = []
    for _ in range(k):
        dp, dv, hidden = dyn(pos, vel, hidden)
        pos = pos + vel * dt + dp
        vel = vel + dv
        ps.append(pos)
    return torch.stack(ps), vel


def tile_check(dyn, pos, tiles):
    with torch.no_grad():
        glob = dyn.accel(pos)
        r = dyn.neighbor_radius
        x = pos[:, 0]
        edges = torch.linspace(x.min().item(), x.max().item() + 1e-3, tiles + 1)
        out = torch.zeros_like(glob)
        for t in range(tiles):
            own = (x >= edges[t]) & (x < edges[t + 1])
            keep = (x >= edges[t] - r) & (x < edges[t + 1] + r)
            sub = pos[keep]
            a = dyn.accel(sub)
            out[own] = a[own[keep]]
    return (out - glob).abs().max().item(), glob.abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sizes", default="50,200,800,3200,12800")
    ap.add_argument("--train-min", type=int, default=20)
    ap.add_argument("--train-max", type=int, default=100)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--cutoff", type=float, default=100.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/gravity_tiling_test.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    dmax = trained_dmax((args.train_min, args.train_max), args.steps, 4738, args.eps, args.dt, dev)
    print("trained max pair distance", round(dmax, 2), flush=True)
    sd = torch.load(args.checkpoint, map_location=dev)
    variants = {}
    for name, radius, clamp in (("cutoff", args.cutoff, False), ("allpairs_raw", 1e9, False), ("allpairs_clamp", 1e9, True)):
        dyn = CentralForceDynamics(dt=args.dt, neighbor_radius=radius).to(dev)
        dyn.load_state_dict(sd)
        if clamp:
            dyn.force = nn.Sequential(ClampLog(float(np.log(dmax))), dyn.force)
        variants[name] = dyn
    res = {"dmax_trained": dmax, "args": vars(args), "sizes": {}}
    for n in [int(s) for s in args.sizes.split(",")]:
        row = {}
        for seed in (9000, 12000):
            rng = np.random.default_rng(seed + n)
            for s in range(args.scenes):
                p, v = gs.init_bodies(n, rng, scale=True)
                P, V = gs.rollout_torch(p, v, args.steps, dev, dt=args.dt, eps=args.eps)
                p0 = torch.tensor(P[0], dtype=torch.float32, device=dev)
                v0 = torch.tensor(V[0], dtype=torch.float32, device=dev)
                tp = torch.tensor(P[1:], dtype=torch.float32, device=dev)
                e_true = energy(torch.tensor(P[-1], device=dev), torch.tensor(V[-1], device=dev), args.eps)
                cv = [np.linalg.norm(P[0] + V[0] * args.dt * (t + 1) - P[t + 1], axis=1).mean() for t in range(args.steps)]
                cloud = float(np.linalg.norm(P[0] - P[0].mean(0), axis=1).mean())
                row.setdefault("cloud_radius", []).append(cloud)
                row.setdefault("constvel", []).append([cv[i] for i in (4, 9, 19)])
                row.setdefault("energy_true", []).append(e_true)
                for name, dyn in variants.items():
                    with torch.no_grad():
                        ps, vf = rollout_model(dyn, p0, v0, args.steps, args.dt)
                    err = (ps - tp).norm(dim=-1).mean(dim=1).cpu().numpy()
                    e_m = energy(ps[-1].double(), vf.double(), args.eps)
                    row.setdefault(name, []).append([float(err[i]) for i in (4, 9, 19)])
                    row.setdefault(name + "_energy", []).append(e_m)
                    del ps
                torch.cuda.empty_cache()
        summ = {k: np.mean(v, axis=0).tolist() if isinstance(v[0], list) else float(np.mean(v)) for k, v in row.items()}
        res["sizes"][str(n)] = summ
        print(n, "cloud", round(summ["cloud_radius"], 1), "constvel", np.round(summ["constvel"], 4),
              {k: np.round(summ[k], 4).tolist() for k in variants}, "E true", round(summ["energy_true"], 1),
              {k: round(summ[k + "_energy"], 1) for k in variants}, flush=True)
    rng = np.random.default_rng(4738)
    p, _ = gs.init_bodies(6400, rng, scale=True)
    diff, mx = tile_check(variants["cutoff"], torch.tensor(p, dtype=torch.float32, device=dev), 4)
    res["tile_check"] = {"n": 6400, "tiles": 4, "max_abs_diff": diff, "max_abs_accel": mx}
    print("tile check (4 strips, halo=cutoff) max abs diff", diff, "max |a|", mx, flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
