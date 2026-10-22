"""Core drift speed from escape_roll.py outputs: own-centre (shrinking sphere) of each arm vs truth at the snapshot ticks.
drift = least-squares slope (units/tick) of |c_arm - c_truth| over snapshots t in [100, 1000] (core counted only while >= half of the truth's bodies within 10 units).
Usage: python scripts/armA_drift.py results/armA/n2e4_s4738 [arm ...]"""
import os
import sys

import numpy as np


def centre(pos, iters=(30.0, 15.0, 8.0, 5.0, 5.0)):
    c = np.median(pos, 0)
    for R in iters:
        sel = np.linalg.norm(pos - c, axis=-1) < R
        if sel.sum() > 20:
            c = pos[sel].mean(0)
    return c


def series(path):
    z = np.load(path)
    out = []
    for t, p in zip(z["snap_t"], z["snap_pos"]):
        c = centre(p)
        out.append((int(t), c, int((np.linalg.norm(p - c, axis=-1) < 10).sum())))
    return out


def main():
    d = sys.argv[1]
    names = sys.argv[2:] or sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz") and f != "truth.npz")
    tr = {t: (c, n) for t, c, n in series(f"{d}/truth.npz")}
    for n in names:
        rows = [(t, c, k) for t, c, k in series(f"{d}/{n}.npz") if t >= 100 and t in tr]
        ok = [(t, np.linalg.norm(c - tr[t][0])) for t, c, k in rows if k >= 0.5 * tr[t][1]]
        if len(ok) < 3:
            print(f"{n:6s} core lost")
            continue
        t = np.array([a for a, _ in ok], float)
        e = np.array([b for _, b in ok])
        slope = np.polyfit(t, e, 1)[0]
        last = ok[-1]
        print(f"{n:6s} drift {slope:.4f}/tick  |dc| at t={last[0]} {last[1]:.2f}  intact to t={last[0]} of {rows[-1][0]}")


if __name__ == "__main__":
    main()
