import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out, scene = sys.argv[1], sys.argv[2], int(sys.argv[3])
steps = [0, 5, 10, 20, 50, 100]
arms = {a: np.load(f"{root}/ef_{a}/E_ms_kp_pot_v_g128_traj.npz") for a in ("sq", "euclid")}
rows = [("truth", arms["sq"]["truth"][scene]), ("model A", arms["sq"]["model"][scene]), ("model B", arms["euclid"]["model"][scene])]
fig, ax = plt.subplots(3, len(steps), figsize=(2.6 * len(steps), 8))
lim = np.abs(rows[0][1]).max() * 1.05
for r, (name, tr) in enumerate(rows):
    for c, s in enumerate(steps):
        a = ax[r, c]
        a.scatter(tr[s][:, 0], tr[s][:, 1], s=4)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xticks([])
        a.set_yticks([])
        if r == 0:
            a.set_title(f"step {s}")
        if c == 0:
            a.set_ylabel(name)
plt.tight_layout()
plt.savefig(out, dpi=80)
