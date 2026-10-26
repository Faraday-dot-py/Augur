"""Model grid force vs exact direct-sum force by distance from the grid edge, from a scatter_bh_1b run with --snaps --grid-dump (all bodies in snaps, i.e. n <= nsub).

Exact a(x) = sum_j d_j / (|d_j|^2 + eps^2)^1.5 at the 128x128 cell centres (unit masses, all bodies including those outside the box). Model |a| is grids[:, 1].
edge distance = 32 - max(|x|, |y|) of the cell centre.

Usage: python3 scripts/bh_edge_force_check.py results/scatter_bh_edge_new_1e6 results/scatter_bh_edge_new_1e6_edgecheck.json [ticks]
"""
import json
import sys

import numpy as np
import torch

stem, out = sys.argv[1], sys.argv[2]
ticks = [int(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 11, 40, 100]
grids = np.load(stem + "_grids.npy")
snaps = np.load(stem + "_snaps.npz")["pos"]
dev = torch.device("cuda")
G, h, half, eps = 128, 0.5, 32.0, 0.5
c = (torch.arange(G, device=dev, dtype=torch.float32) + 0.5) * h - half
cx = torch.stack([c[None, :].expand(G, G), c[:, None].expand(G, G)], -1).reshape(-1, 2)
edge = (half - cx.abs().max(-1).values).cpu().numpy()
bands = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 8), (8, 16), (16, 33)]
res = {}
for t in ticks:
    src = torch.from_numpy(snaps[t]).to(dev).float()
    ex = torch.zeros_like(cx)
    for s in range(0, cx.shape[0], 64):
        d = src[None] - cx[s:s + 64, None]
        ex[s:s + 64] = (d * ((d ** 2).sum(-1, keepdim=True) + eps ** 2) ** -1.5).sum(1)
    ex = ex.norm(dim=-1).cpu().numpy()
    mo = grids[t, 1].reshape(-1)
    rows = {}
    for lo, hi in bands:
        k = (edge >= lo) & (edge < hi)
        r = mo[k] / ex[k]
        rows[f"{lo}-{hi}"] = dict(model_max=float(mo[k].max()), exact_max=float(ex[k].max()), ratio_of_means=float(mo[k].mean() / ex[k].mean()),
                                  median_ratio=float(np.median(r)), max_ratio=float(r.max()))
    res[t] = rows
    print(f"tick {t}")
    for b, r in rows.items():
        print(f"  edge dist {b:>6}: model max {r['model_max']:.4g} exact max {r['exact_max']:.4g} mean ratio {r['ratio_of_means']:.3f} median ratio {r['median_ratio']:.3f} max ratio {r['max_ratio']:.2f}", flush=True)
json.dump(res, open(out, "w"), indent=1)
