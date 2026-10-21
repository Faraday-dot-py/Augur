"""Find non-finite bodies in the chunked galaxy IC at large N. Usage: PYTHONPATH=. python scripts/galaxy_ic_nan.py --n 1000000000"""
import argparse

import torch

from scripts import galaxy_ic as gi

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--chunk", type=int, default=50_000_000)
args = ap.parse_args()
dev = torch.device("cuda")
P = gi.PRESETS["g1"]
tab = gi.tables(P, dev)
g = torch.Generator(device=dev)
g.manual_seed(4738)
tot = 0
for s in range(0, args.n, args.chunk):
    m = min(args.chunk, args.n - s)
    p, v = gi.sample_chunk(m, P, g, dev, tab)
    bad = ~torch.isfinite(p).all(-1) | ~torch.isfinite(v).all(-1)
    nb = int(bad.sum())
    tot += nb
    if nb:
        r = p.norm(dim=-1)
        print(s, nb, "pos", p[bad][:5].tolist(), "vel", v[bad][:5].tolist(), "r", r[bad][:5].tolist(), flush=True)
print("total bad", tot)
