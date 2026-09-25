"""Tables and log-log fits from results/scaling_*.json.

Usage: python scripts/scaling_summarize.py
"""
import glob
import json

import numpy as np


def fit(ns, ys):
    ns, ys = np.array(ns, float), np.array(ys, float)
    if len(ns) < 2:
        return float("nan"), float("nan")
    p = np.polyfit(np.log(ns), np.log(ys), 1)
    return float(p[0]), float(np.exp(p[1]))


def main():
    rows = []
    for f in sorted(glob.glob("results/scaling_*.json")):
        d = json.load(open(f))
        if f.split("scaling_")[1].split(".")[0] in ("a", "b", "c_uni", "c_fly", "t10k"):
            rows += d["rows"]
    rows.sort(key=lambda r: (r["state"], r["N"]))
    for state in sorted({r["state"] for r in rows}):
        rs = [r for r in rows if r["state"] == state]
        print(f"\n## {state}")
        print("N | est s | est cost | est rel_l2 | lam | est GB | geo.35 s | geo.35 cost | geo.35 rel_l2 | geo_m theta | geo_m s | geo_m cost | geo_m GB | exact s | bh s | mesh (grid,s,rel_l2)")
        for r in rs:
            m = r.get("mesh", [])
            mb = min([x for x in m if "rel_l2" in x], key=lambda x: x["rel_l2"], default=None)
            print(" | ".join(str(x) for x in [
                r["N"], f"{r['est']['median_s']:.3f}", f"{r['est']['cost']:.1f}", f"{r['est']['rel_l2']:.4f}", f"{r['est']['lam']:.2f}", f"{r['est']['peak_gb']:.2f}",
                f"{r['geo035']['median_s']:.3f}", f"{r['geo035']['cost']:.1f}", f"{r['geo035']['rel_l2']:.4f}", f"{r['geo_matched']['theta']:.3f}",
                f"{r['geo_matched']['median_s']:.3f}", f"{r['geo_matched']['cost']:.1f}", f"{r['geo_matched']['peak_gb']:.2f}",
                f"{r['exact']['median_s']:.3f}" if "exact" in r else "-", f"{r['bh']['median_s']:.3f}" if "bh" in r and "median_s" in r["bh"] else "-",
                f"{mb['grid']},{mb['median_s']:.3f},{mb['rel_l2']:.3f}" if mb else "-"]))
        for key, sub in (("est", "median_s"), ("est", "peak_gb"), ("geo035", "median_s"), ("geo035", "peak_gb"), ("geo_matched", "median_s"), ("exact", "median_s")):
            pts = [(r["N"], r[key][sub]) for r in rs if key in r and sub in r[key]]
            if len(pts) >= 2:
                print(f"fit {key}.{sub}: exponent {fit(*zip(*pts))[0]:.3f} (all N), ", end="")
                big = [p for p in pts if p[0] >= 100000]
                if len(big) >= 2:
                    print(f"{fit(*zip(*big))[0]:.3f} (N>=1e5)")
                else:
                    print()
        for key in ("est", "geo035", "geo_matched"):
            c = [r[key]["cost"] for r in rs if key in r]
            if c:
                print(f"cost {key}: min {min(c):.1f} max {max(c):.1f} variation {(max(c) - min(c)) / np.mean(c):.2%}")
        print("breakdown est (s):")
        for r in rs:
            print(r["N"], {k: round(v, 4) for k, v in r["est_breakdown_s"].items()}, "audit1000", round(r["audit_k1000_s"]["median_s"], 4))


if __name__ == "__main__":
    main()
