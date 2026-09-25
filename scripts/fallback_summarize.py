"""Summarise results/fallback_*.json. Usage: python scripts/fallback_summarize.py [glob-prefix ...]"""
import glob
import json
import sys

import numpy as np


def summ(path):
    o = json.load(open(path))
    c, tr, ev, dg = o["cfg"], o["trace"], o["events"], o["diags"]
    tgt = c["target"]
    fb = [e for e in ev if e["event"] == "fallback"]
    pr = [e for e in ev if e["event"] == "probe estimator"]
    pp = [e for e in ev if e["event"] == "probe passed"]
    first_fb = fb[0]["call"] if fb else None
    tv = [(r["step"], r["mode"], r["probing"], r["true"]) for r in tr if r["true"] is not None]
    first_bad = next((s for s, m, p, t in tv if t > 3 * tgt), None)
    def stat(sel):
        v = np.array([t for s, m, p, t in tv if sel(s, m, p)])
        return None if len(v) == 0 else (float(v.mean()), float(np.median(v)), float(v.max()), len(v))
    ff = first_fb if first_fb is not None else 10**9
    est_ok = lambda s, m, p: m == "est" and not p
    r = {"name": c["name"], "first_fb": first_fb, "n_fb": len(fb), "n_probe": len(pr), "n_probe_pass": len(pp),
         "probe_fail": sum(1 for e in fb if e["why"] == "probe failed"), "first_bad": first_bad,
         "before": stat(lambda s, m, p: s < ff), "geo": stat(lambda s, m, p: m == "geo"), "est_after": stat(lambda s, m, p: s >= ff and m == "est" and not p),
         "probe_err": stat(lambda s, m, p: p), "all": stat(lambda s, m, p: True), "est_all": stat(est_ok),
         "frac_est": float(np.mean([r_["mode"] == "est" for r_ in tr])), "cost": float(np.mean([r_["cost"] for r_ in tr])),
         "time_force": float(np.mean([r_["time_s"] for r_ in tr])), "why": [e["why"] for e in fb][:3],
         "over3": float(np.mean([t > 3 * tgt for s, m, p, t in tv])) if tv else None,
         "over1.5": float(np.mean([t > 1.5 * tgt for s, m, p, t in tv])) if tv else None}
    e0 = dg[0]["E"]
    r["dE_max"] = max(abs(d["E"] - e0) / abs(e0) for d in dg)
    r["dE_end"] = (dg[-1]["E"] - e0) / abs(e0)
    r["P_max"] = max(d["P"] / d["sumv"] for d in dg)
    r["steps"] = c["steps"]
    r["wall"] = o["wall_s"]
    return r


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, tuple):
        return f"{x[0]:.4f}/{x[1]:.4f}/{x[2]:.4f} (n={x[3]})"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


if __name__ == "__main__":
    pats = sys.argv[1:] or [""]
    for p in pats:
        for f in sorted(glob.glob(f"results/fallback_{p}*.json")):
            if f.endswith("_ckpt.json"):
                continue
            r = summ(f)
            print(r["name"])
            for k, v in r.items():
                if k != "name":
                    print("   ", k, fmt(v))
