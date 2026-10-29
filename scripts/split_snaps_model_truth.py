"""Split a scatter_bh_large snaps npz (pos = model, truth_pos = exact) into two npz files for render_nbody_lens.py. Usage: python3 scripts/split_snaps_model_truth.py in.npz out_model.npz out_truth.npz"""
import sys

import numpy as np

d = np.load(sys.argv[1])
np.savez(sys.argv[2], pos=d["pos"], steps=d["steps"])
np.savez(sys.argv[3], pos=d["truth_pos"], steps=d["steps"])
print(d["pos"].shape, d["truth_pos"].shape)
