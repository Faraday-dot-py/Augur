import argparse

import numpy as np
import torch

from scripts import scatter_scaling_ops as ops
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling import TiledScatter
from scripts.scatter_tiling_eval import DT, EPS, SUB, load, single_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--sigma", type=float, default=16.0)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--seed", type=int, default=4738)
ap.add_argument("--nt", type=int, default=2)
ap.add_argument("--tile", type=float, default=64.0)
ap.add_argument("--halo", type=float, default=16.0)
a = ap.parse_args()

dev = "cuda"
model = load(a.ckpt, dev)
p0, v0 = make_scene(a.n, a.sigma, 1.0, 128.0, a.seed, 0)
Pt, _ = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), a.steps, DT, SUB, EPS)
pos = torch.tensor(p0, dtype=torch.float32, device=dev)
vel = torch.tensor(v0, dtype=torch.float32, device=dev)
mass = torch.ones(a.n, device=dev)
msk = torch.ones(1, a.n, device=dev)

with torch.no_grad():
    single_setup(model)
    st = (model.init_field(1, dev)[0], None)
    p, v = pos[None], vel[None]
    Ps = [p[0]]
    for _ in range(a.steps):
        p, v, st, _ = model.step(p, v, mass[None], msk, st)
        Ps.append(p[0])
    single_setup(model)
    T = TiledScatter(model, a.nt, a.tile, a.halo, ghosts=True, field_x=True, far=True)
    ts = (T.init_field(dev), None)
    p, v = pos, vel
    Pl, own = [p], []
    for _ in range(a.steps):
        p, v, ts = T.step(p, v, mass, ts)
        Pl.append(p)
    single_setup(model)

np.savez(a.out, exact=Pt.float().cpu().numpy(), single=torch.stack(Ps).cpu().numpy(), tiled=torch.stack(Pl).cpu().numpy(),
         nt=a.nt, tile=a.tile, halo=a.halo, seed=a.seed)
print("done", a.out, flush=True)
