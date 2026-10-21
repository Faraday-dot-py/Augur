"""Windowed reversal statistics (outgoing bodies in the band 28-32 whose axial velocity flips from >1 outward to <-1 inward) from edge_longroll.py npz files.
Usage: python3 scripts/edge_window_stats.py name=file.npz ..."""
import json
import sys

import numpy as np

WINDOWS = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 700), (700, 1000)]
for spec in sys.argv[1:]:
    name, path = spec.split("=")
    d = np.load(path)
    out, rev = d["out"], d["rev"]
    stats = json.loads(str(d["stats"]))
    row = []
    for lo, hi in WINDOWS:
        o, r = int(out[lo:hi].sum()), int(rev[lo:hi].sum())
        row.append(f"{lo}-{hi}: {r}/{o} = {100 * r / max(o, 1):.2f}%")
    fo = {s["t"]: s["frac_out"] for s in stats}
    print(f"{name}: " + " | ".join(row))
    print(f"   frac outside box at t=100/300/400/500/700: " + " / ".join(f"{fo.get(t, float('nan')):.3f}" for t in (100, 300, 400, 500, 700)))
