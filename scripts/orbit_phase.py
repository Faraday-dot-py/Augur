"""Orbits completed vs time from an orbit_<tag>.json binary track: unwrapped angle of (centre A - centre B).
Usage: python scripts/orbit_phase.py results/orbit_X.json [every_steps]
"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
every = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
dt = d["args"]["dt"]
st = np.array(d["track_steps"])
rel = np.array([np.array(t["ca"]) - np.array(t["cb"]) for t in d["tracks"]])
ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
turns = np.abs(ang - ang[0]) / (2 * np.pi)
dg = {g["step"]: g for g in d["diags"]}
for i, s in enumerate(st):
    if s % every == 0:
        t = d["tracks"][i]
        g = min(dg.values(), key=lambda g: abs(g["step"] - s))
        print(f"step {s:7d} t {s * dt:6.0f} orbits {turns[i]:.2f} sep {t['sep']:7.1f} a_r4 {t['a_r4']:.3f} a_r8 {t['a_r8']:.3f} a_own {t['a_own']:.3f} b_r8 {t['b_r8']:.3f} | E {g['E']:.4g} Epos {g['frac_E_pos']:.4f} L {g['L']:.5g} P {g['P']:.2e}")
print("total orbits", round(float(turns[-1]), 2))
