"""Compare outgoing-shell front radius (outermost r with surface density/N > 2e-5), r50 collapse minimum tick, and fan-ridge positions vs the axisymmetric reference.
Usage: python scripts/bh_wave_fronts.py <data_dir> <ref_dir> <scene> <ref_stem>"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from bh_wave_analyze import AREA, RLIN, RMID, frac, front  # noqa: E402


def main():
    d, rd, scene, ref = sys.argv[1:5]
    arms = {a: np.load(f"{d}/{scene}_{a}.npz")["rad_lin"].astype(float) for a in ["truth4", "pm128", "pm256", "pm128s4", "model_old", "model_new"] if __import__("os").path.exists(f"{d}/{scene}_{a}.npz")}
    arms["ref"] = np.load(f"{rd}/{ref}.npz")["rad_lin"].astype(float)
    ts = [20, 30, 40, 50, 60, 70]
    print("front r at ticks", ts)
    for a, c in arms.items():
        fr = front(c, 2e-5)
        r50 = np.array([np.interp(0.5, np.cumsum(frac(c)[t]), RLIN[1:]) for t in range(c.shape[0])])
        rr = np.array([np.interp(0.9, np.cumsum(frac(c)[t]), RLIN[1:]) for t in range(c.shape[0])])
        print(f"{a:10s} front {[round(fr[t], 1) for t in ts]} r50 min tick {int(np.argmin(r50[:30]))} r50min {r50[:30].min():.2f} r90 at 10,20,30 {[round(rr[t], 1) for t in (10, 20, 30)]}")
    f = {a: frac(c) / AREA for a, c in arms.items()}
    print("fan-ridge radii (local maxima of smoothed profile, 6<r<22) at tick 60/100:")
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    for a, c in arms.items():
        out = []
        for t in (60, 100):
            p = gaussian_filter1d(f[a][t], 2)
            lo, hi = np.searchsorted(RMID, 6), np.searchsorted(RMID, 22)
            pk, _ = find_peaks(np.log10(p[lo:hi] + 1e-9), prominence=0.03)
            out.append([round(float(RMID[lo + k]), 1) for k in pk])
        print(f"{a:10s} {out}")


if __name__ == "__main__":
    main()
