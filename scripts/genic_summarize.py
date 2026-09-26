"""Summaries for genic: matched-error costs (static, retrain, scaling) and rollout tables.

Usage: python3 scripts/genic_summarize.py static|retrain|scaling|rollout <json files or tag prefix>
"""
import glob
import json
import sys

import numpy as np


def cost_at(rows, key, target):
    pts = sorted((r[key], r["cost"]) for r in rows)
    e, c = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    t = np.log(target)
    if t < e.min() or t > e.max():
        return float("nan")
    return float(np.exp(np.interp(t, e, c)))


def merged(files):
    out = {}
    for f in files:
        d = json.load(open(f))
        for k, v in d.items():
            if isinstance(v, dict):
                out.setdefault(k, {}).update(v)
    return out


def row(name, s, est_rows):
    cells = []
    for key, tg in (("rel_l2", 0.01), ("rel_l2", 0.02), ("abs_p99", 0.03)):
        g, b, e = cost_at(s["geo"], key, tg), cost_at(s["bh"], key, tg) if s["bh"] else float("nan"), cost_at(est_rows, key, tg)
        cells.append(f"{g:6.0f} {b:6.0f} {e:6.0f} {e / g:5.2f}")
    return f"{name:16s} | " + " | ".join(cells)


def static(files):
    d = merged(files)
    print("state            | rel_l2 0.01: geo  bh  est est/geo | rel_l2 0.02 | abs_p99 0.03")
    for k, s in d["static"].items():
        print(row(k, s, s["est"]["flyby"]))
    print("calibration (flyby-trained head, fresh pairs): state r2 bias cover_q90 under_frac")
    for k, c in d["calib"].items():
        print(f"{k:16s} {c['r2']:.3f} {c['bias']:+.3f} {c['cover_q90']:.3f} {c['frac_mean_underestimate']:.3f} n={c['n']}")
    if "retrain" in d:
        print("retrain (test at step 1000): est/geo cost ratio at rel_l2 0.01 / 0.02, abs_p99 0.03 per head; calib r2/bias/cov")
        for k, s in d["retrain"].items():
            for hn, rows in s["est"].items():
                r = [cost_at(rows, key, tg) / cost_at(s["geo"], key, tg) for key, tg in (("rel_l2", 0.01), ("rel_l2", 0.02), ("abs_p99", 0.03))]
                c = s["calib"][hn]
                print(f"{k:14s} {hn:12s} cost {cost_at(rows, 'rel_l2', 0.01):6.0f} ratios {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} | r2 {c['r2']:.3f} bias {c['bias']:+.3f} q90 {c['cover_q90']:.3f}")


def scaling(files):
    d = merged(files)
    for k, s in d.items():
        e = s["est"]["flyby"]
        t3 = [r for r in e if r["tol"] == 1e-3][0]
        g35 = min(s["geo"], key=lambda r: abs(r["theta"] - 0.35))
        c = s["calib"]
        au = {x["mode"]: x for x in s["audited"]}
        print(f"{k:16s} tol1e-3 cost {t3['cost']:.0f} rel_l2 {t3['rel_l2']:.4f} p99abs {t3['abs_p99']:.3f} | geo.35 cost {g35['cost']:.0f} rel {g35['rel_l2']:.4f} | r2 {c['r2']:.3f} bias {c['bias']:+.3f} q90 {c['cover_q90']:.3f} | "
              f"adaptive cost {au['adaptive']['cost_last10']:.0f} rel {au['adaptive']['rel_l2']:.4f} lam {au['adaptive']['lam']:.2f} now {au['adaptive']['final_now']} ev {len(au['adaptive']['events'])} | geo_audit cost {au['geo_audit']['cost_last10']:.0f} rel {au['geo_audit']['rel_l2']:.4f}")


def rollout(prefix):
    for f in sorted(glob.glob(f"results/rollout_{prefix}_*.json")):
        if f.endswith("_ckpt.json"):
            continue
        d = json.load(open(f))
        dg = d["diags"]
        e0, e1 = dg[0]["E"], dg[-1]["E"]
        Lm = max(abs(x["L"] - dg[0]["L"]) for x in dg)
        Pm = max(x["P"] for x in dg)
        vr = d.get("vs_ref", [])
        last = vr[-1] if vr else None
        fo = d.get("force")
        s = f"{f.split('rollout_')[1][:-5]:34s} dE/|E| {(e1 - e0) / abs(e0):+.2e} maxP {Pm:.2e} max|dL| {Lm:.2e} L0 {dg[0]['L']:.1e} step {d['step_time_mean']:.3f}s"
        if last:
            s += f" | dens_corr {last['density_corr']:.4f} rq_ratio {[round(x, 3) for x in last['rq_ratio']]} dx_med {last['dx_median']:.2f}"
        if fo:
            s += f" | cost {fo['mean_cost']:.0f} est_frac {fo['mode_fraction_est']:.2f} fallbacks {sum(1 for x in fo['events'] if x['event'] == 'fallback')}"
            au = [a[2] for a in fo["audits"]]
            s += f" audit mean {np.mean(au):.4f} max {np.max(au):.4f}"
        print(s)


if __name__ == "__main__":
    m, rest = sys.argv[1], sys.argv[2:]
    {"static": static, "retrain": static, "scaling": scaling}.get(m, lambda r: rollout(r[0]))(rest)
