"""Linear interpolation of body positions between saved snapshots of an npz (pos, steps): k-1 extra frames per pair.

Usage: python3 scripts/interp_snaps.py in_snaps.npz out_snaps.npz [k=3]
"""
import sys

import numpy as np

d = np.load(sys.argv[1])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
pos, steps = d["pos"], d["steps"].astype(np.float64)
out = np.empty(((len(pos) - 1) * k + 1,) + pos.shape[1:], dtype=pos.dtype)
st = np.empty(len(out))
for i in range(len(pos) - 1):
    for j in range(k):
        w = j / k
        out[i * k + j] = (1 - w) * pos[i] + w * pos[i + 1]
        st[i * k + j] = (1 - w) * steps[i] + w * steps[i + 1]
out[-1], st[-1] = pos[-1], steps[-1]
np.savez(sys.argv[2], pos=out, steps=st.round().astype(int))
