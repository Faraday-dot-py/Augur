"""Rollout dump for the wide-scene video: exact fp64 | tiled+far with the shipped kernel | tiled+far with a replacement kernel.
Usage: PYTHONPATH=. python scripts/scatter_far_video.py --ckpt kern/E_frozen.pt --kernel file:kern/r1_mlp.pt --out results/farvideo_gauss.npz [--kind gauss|two] [--steps 100]
Render locally with scripts/render_far_video.py.
"""
import argparse

import numpy as np
import torch

from scripts import scatter_far as sfar
from scripts import scatter_far_eval as fe
from scripts import scatter_scaling_ops as ops
from scripts.scatter_tiling import TiledScatter
from scripts.scatter_tiling_eval import DT, EPS, SUB, load

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--kernel", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--kind", default="gauss")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--seed", type=int, default=4738)
ap.add_argument("--nt", type=int, default=4)
ap.add_argument("--n", type=int, default=4800)
ap.add_argument("--sigma", type=float, default=32.0)
a = ap.parse_args()
dev = torch.device("cuda")
fe.set_scene(a.nt, a.n, a.sigma, a.kind)
m = load(a.ckpt, dev)
p0, v0 = fe.scene(a.seed)
Pt, _ = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), a.steps, DT, SUB, EPS)
pos = torch.tensor(p0, dtype=torch.float32, device=dev)
vel = torch.tensor(v0, dtype=torch.float32, device=dev)
mass = torch.ones(a.n, device=dev)
out = {"exact": Pt.float().cpu().numpy()}
for name, spec in (("old", "old"), ("fixed", a.kernel)):
    sfar.install_spec(m, spec)
    ops.retarget(m, round(fe.L / 0.5), fe.L)
    T = TiledScatter(m, a.nt, 64, 16, ghosts=True, field_x=True, far=True)
    _, P, _ = fe.run_tiled(T, pos, vel, mass, a.steps)
    out[name] = P.float().cpu().numpy()
    print(name, "err@20", float((P[20] - Pt[20]).norm(dim=-1).mean()), "err@end", float((P[-1] - Pt[-1]).norm(dim=-1).mean()), flush=True)
    sfar.install(m, None)
np.savez(a.out, nt=a.nt, tile=64.0, halo=16.0, seed=a.seed, kind=a.kind, **out)
print("done", a.out)
