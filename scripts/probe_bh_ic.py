"""Dry-run probe of the orbit_bh.py blackhole initial condition (GPU): measured virial dispersion per component at several --ratio,
initial |v| quantiles, central potential depth / escape speed, learned_rel vs analytic acceleration on a body subsample, and learned force timing.

Usage: PYTHONPATH=. python scripts/probe_bh_ic.py [--n 100000] [--sigma 10]
"""
import argparse
import time

import torch

from scripts import adaptive_oracle as ao
from scripts import est_train
from scripts import kernels
from scripts import nbody_ic
from scripts import orbit_bh as ob

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--sigma", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=4738)
args = ap.parse_args()
dev = torch.device("cuda")
torch.set_grad_enabled(False)
kernel = est_train.make_kernel("analytic", dev)
kernels.current = kernel
torch.manual_seed(args.seed)
gen = nbody_ic._gen(dev, args.seed)
pos = torch.randn(args.n, 2, generator=gen, device=dev, dtype=torch.float64) * args.sigma
q = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], device=dev, dtype=torch.float64)
for ratio in (0.5, 0.3, 0.2, 0.1, 0.05):
    sv = nbody_ic.virial_sigma(pos, kernel, ratio)
    v = torch.randn(args.n, 2, generator=torch.Generator(device=dev).manual_seed(1), device=dev, dtype=torch.float64) * sv
    print(f"ratio {ratio} sigma_v/component {sv:.2f} |v| q50/90/99/99.9/max {[round(x, 1) for x in v.norm(dim=1).quantile(q).tolist()]}", flush=True)
phi = ob.potentials(pos)
vesc = (2 * (-phi).clamp(min=0)).sqrt()
print(f"phi min {float(phi.min()):.1f} median {float(phi.median()):.1f}; vesc q50/90/99/99.9/max {[round(x, 1) for x in vesc.quantile(q).tolist()]}", flush=True)
a = argparse.Namespace(kernel="learned_rel", checkpoint="checkpoints/gravity_relativistic_central.pt", lchunk=512, force="exact")
force = ob.make_force(a, args.n, dev)
torch.cuda.synchronize()
t0 = time.time()
al = force(pos)
torch.cuda.synchronize()
print(f"learned force eval {time.time() - t0:.2f} s", flush=True)
idx = torch.randperm(args.n, device=dev)[:4000]
aa = ao.exact_accel(pos, idx)
ratio = (al[idx] * aa).sum(1) / (aa * aa).sum(1)
r = (pos[idx] - pos.median(0).values).norm(dim=1)
print(f"learned/analytic radial projection q10/50/90 {[round(x, 3) for x in ratio.quantile(torch.tensor([0.1, 0.5, 0.9], device=dev, dtype=torch.float64)).tolist()]}", flush=True)
for lo, hi in ((0, 3), (3, 10), (10, 20), (20, 100)):
    m = (r >= lo) & (r < hi)
    if m.any():
        rel = (al[idx][m] - aa[m]).norm(dim=1) / aa[m].norm(dim=1)
        print(f"r in [{lo},{hi}) n {int(m.sum())} rel err median {float(rel.median()):.4f} |a| median {float(aa[m].norm(dim=1).median()):.1f}", flush=True)
