"""Analysis of results/bhwave/*.npz (bh_wave_run.py) and ref_*.npz (bh_shell_ref.py): space-time maps, front slope, caustic-front period, profile distance to truth,
ring contrast vs N, energy / virial / L / spin tables.

Usage: python scripts/bh_wave_analyze.py <data_dir> <ref_dir> <out_dir>
Writes PNGs and summary.json to out_dir and prints the tables used in docs/debugging/bh-wave-structures.md.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

RLIN = np.arange(0, 48.01, 0.25)
RMID = 0.5 * (RLIN[1:] + RLIN[:-1])
AREA = np.pi * (RLIN[1:] ** 2 - RLIN[:-1] ** 2)
ARMS = ["truth4", "truth1", "pm128", "pm256", "pm128s4", "model_old", "model_new"]


def frac(c):
    return c / c.sum(1, keepdims=True).clip(min=1)


def front(c, thr=2e-5):
    f = frac(c) / AREA
    out = np.full(c.shape[0], np.nan)
    for t in range(c.shape[0]):
        k = np.nonzero(f[t] > thr)[0]
        out[t] = RMID[k.max()] if len(k) else np.nan
    return out


def front_slope(c):
    fr = front(c)
    t = np.arange(len(fr))
    s = (t >= 25) & (t <= 55) & np.isfinite(fr) & (fr < 44)
    return float(np.polyfit(t[s], fr[s], 1)[0]) if s.sum() > 4 else np.nan, fr


def emd(c, ref, t):
    a, b = np.cumsum(frac(c)[t]), np.cumsum(frac(ref)[t])
    return float(np.abs(a - b).sum() * 0.25)


def annulus_period(c, r0=8.0, r1=14.0, t0=60):
    k = (RMID > r0) & (RMID < r1)
    s = frac(c)[:, k].sum(1)[t0:]
    s = s - gaussian_filter1d(s, 12, mode="nearest")
    ps = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=1024)) ** 2
    fr = np.fft.rfftfreq(1024, 1.0)
    j = np.argmax(ps[8:]) + 8
    return float(1.0 / fr[j]), float(ps[j] / ps[8:].sum())


def contrast(c, ticks, r0=3.0, r1=25.0):
    k = (RMID > r0) & (RMID < r1)
    raw, poi = [], []
    for t in ticks:
        x = c[t].astype(float)
        s = gaussian_filter1d(x, 6.0, mode="nearest")
        rel = (x[k] - s[k]) / np.maximum(s[k], 1e-9)
        raw.append(np.mean(rel ** 2))
        poi.append(np.mean(1.0 / np.maximum(s[k], 1e-9)))
    return float(np.sqrt(max(np.mean(raw) - np.mean(poi), 0))), float(np.sqrt(np.mean(raw))), float(np.sqrt(np.mean(poi)))


def load(d, s, a):
    p = f"{d}/{s}_{a}.npz"
    return np.load(p) if os.path.exists(p) else None


def main():
    d, rd, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    summ = {}
    scenes = {"n1e4_s4738": 1e4, "n1e4_s9000": 1e4, "n3e4_s4738": 3e4, "n1e5_s4738": 1e5, "n1e6_s4738": 1e6, "n1e7_s4738": 1e7}
    refmap = {1e4: "ref_n1e4_m2e5_s4", 3e4: None, 1e5: "ref_n1e5_m2e5_s4", 1e6: "ref_n1e6_m2e4_s4", 1e7: None}
    for s, n in scenes.items():
        arms = {a: load(d, s, a) for a in ARMS}
        arms = {a: z for a, z in arms.items() if z is not None}
        if not arms:
            continue
        refp = refmap.get(n)
        if refp and os.path.exists(f"{rd}/{refp}.npz"):
            arms["ref"] = np.load(f"{rd}/{refp}.npz")
        if "truth4" not in arms and n >= 1e6:
            arms["ref"] = np.load(f"{rd}/{'ref_n1e6_m2e4_s4' if n == 1e6 else 'ref_n1e8_m2e4_s4'}.npz")
        base = arms.get("truth4", arms.get("ref"))["rad_lin"].astype(float)
        res = {}
        fig, ax = plt.subplots(2, 4, figsize=(17, 8))
        for a, axx in zip(arms, ax.ravel()):
            c = arms[a]["rad_lin"].astype(float)
            sl, fr = front_slope(c)
            per, pw = annulus_period(c)
            cex, craw, cpoi = contrast(c, range(80, 151, 5))
            res[a] = {"front_slope": sl, "period_ticks": per, "period_power_frac": pw, "contrast_excess": cex, "contrast_raw": craw, "poisson": cpoi,
                      "emd_vs_base": {t: emd(c, base, t) for t in (25, 50, 100, 150)},
                      "r50": {t: float(np.interp(0.5, np.cumsum(frac(c)[t]), RLIN[1:])) for t in (25, 50, 100, 150)}}
            axx.imshow(np.log10(frac(c).T / AREA[:, None] + 1e-7), origin="lower", aspect="auto", extent=[0, c.shape[0] - 1, 0, 48], vmin=-5, vmax=-1.5)
            axx.set_title(f"{s} {a}", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{out}/spacetime_{s}.png", dpi=70)
        plt.close()
        summ[s] = res
        print("==", s)
        for a, r in res.items():
            print(f"{a:10s} slope {r['front_slope']:.3f} period {r['period_ticks']:.1f} ({r['period_power_frac']:.2f}) contrast excess {r['contrast_excess']:.3f} raw {r['contrast_raw']:.3f} poi {r['poisson']:.3f} "
                  f"emd {[round(v, 2) for v in r['emd_vs_base'].values()]} r50 {[round(v, 2) for v in r['r50'].values()]}")
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
