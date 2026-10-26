"""Small per-tick extract of a scatter_bh_1b --hist --grid-dump run for the four-panel figure.

Usage: python3 scripts/bh_edge_panels_extract.py results/<tag> results/<tag>_ticks.npz [ticks]
"""
import sys

import numpy as np

stem = sys.argv[1]
ticks = [int(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 11, 40, 100]
h = np.load(stem + "_hist.npz")
g = np.load(stem + "_grids.npy")
np.savez_compressed(sys.argv[2], ticks=np.array(ticks), full=h["full"][ticks], grids=g[ticks], n=h["n"], box_half=h["box_half"])
