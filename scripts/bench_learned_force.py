"""Accuracy and time of the tabulated learned-kernel force (orbit_bh.learned_table_fn) vs direct MLP evaluation. Prints relative error of f(d) on log-uniform d in [1e-6, 300],
and time + relative error of the full exact all-pairs force at N bodies on a Gaussian sigma-10 cloud (direct eval on the first 2048 targets only).
"""
import sys
import time

import torch

from model.central_force import CentralForceDynamics
from scripts import gravity_1b as g1
from scripts import orbit_bh

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
dev = torch.device("cuda")
torch.set_grad_enabled(False)
dyn = CentralForceDynamics(dt=0.1).to(dev)
dyn.load_state_dict(torch.load("checkpoints/gravity_central_v1.pt", map_location=dev))
dyn.eval()
direct, table = g1.model_force(dyn), orbit_bh.learned_table_fn(dyn, dev)
d = torch.exp(torch.rand(2_000_000, 1, device=dev) * (torch.log(torch.tensor(300.0)) - torch.log(torch.tensor(1e-6))) + torch.log(torch.tensor(1e-6)))
a, b = direct(d), table(d)
rel = (a - b).abs() / a.abs().clamp(min=1e-12)
print("f(d) table vs direct: max abs", float((a - b).abs().max()), "max |f|", float(a.abs().max()), "median rel", float(rel.median()), "p99.9 rel", float(rel.quantile(0.999)), flush=True)
pos = torch.randn(n, 2, device=dev) * 10.0
for chunk in (256, 512, 1024):
    torch.cuda.synchronize()
    t = time.time()
    acc = orbit_bh.learned_accel(pos, table, chunk)
    torch.cuda.synchronize()
    print(f"table exact force N={n} chunk {chunk}: {time.time() - t:.2f}s", flush=True)
p = pos[:2048]
rel_pos = pos[None] - p[:, None]
dd = (rel_pos ** 2).sum(-1, keepdim=True).add(1e-12).sqrt()
ref = (direct(dd) * rel_pos / dd).sum(1).double()
print("full force vs direct on 2048 targets: rel_l2", float((acc[:2048] - ref).norm() / ref.norm()), flush=True)
