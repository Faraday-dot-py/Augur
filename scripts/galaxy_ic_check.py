"""IC sanity for scripts/galaxy_ic.py at N=1e5: exact softened radial force vs the enclosed-mass circular speed, mean azimuthal speed, virial ratio, bound fraction. Usage: PYTHONPATH=. python scripts/galaxy_ic_check.py --preset g1 --out results/galaxy/ic_check_g1.json"""
import argparse
import json

import torch

from scripts import galaxy_ic as gi
from scripts import scatter_scaling_ops as ops
from scripts.scatter_galaxy import potential

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="g1")
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--c", type=float, default=100.0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = torch.device("cuda")
P = gi.PRESETS[args.preset]
pos, p = gi.make_galaxy(args.n, P, 4738, args.n, args.c, dev)
m = gi.total_mass(P) / args.n
v = p / (1 + (p ** 2).sum(-1, keepdim=True) / args.c ** 2).sqrt()
a = ops.exact_acc(pos.double(), gi.EPS, torch.full((args.n,), m, device=dev, dtype=torch.float64)).float()
r = pos.norm(dim=-1)
er = pos / r[:, None]
et = torch.stack([-er[:, 1], er[:, 0]], 1)
ar = -(a * er).sum(-1)
vt = (v * et).sum(-1)
vr = (v * er).sum(-1)
edges = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (r >= lo) & (r < hi)
    if int(k.sum()) < 20:
        continue
    rm = r[k].mean()
    rows.append(dict(r_lo=lo, r_hi=hi, count=int(k.sum()), vc_model=float(gi.vcirc(rm, P)), vc_force=float((ar[k].mean() * rm).clamp(min=0).sqrt()), vphi_mean=float(vt[k].mean()),
                     sig_r=float(vr[k].std()), sig_phi=float(vt[k].std())))
ke = float(0.5 * m * (v ** 2).sum())
pot = potential(pos, pos, m)
e_i = 0.5 * (v ** 2).sum(-1) + pot
pe = 0.5 * m * float(pot.sum())
res = dict(rows=rows, ke=ke, pe=pe, virial=2 * ke / abs(pe), bound=float((e_i < 0).float().mean()), vmax_over_c=float(v.norm(dim=-1).max() / args.c), lz=float(m * (pos[:, 0] * v[:, 1] - pos[:, 1] * v[:, 0]).sum()))
json.dump(res, open(args.out, "w"), indent=1)
for row in rows:
    print(row)
print({k: v for k, v in res.items() if k != "rows"})
