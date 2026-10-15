"""Summarize the energy-similarity loss test: losstest JSONs (means over eval seeds 9100/9200/9300 per family ab/cl, incl. esim = |E_model-E_truth|/(K0+|U0|))
and scatter_conservation.py npz dirs (per scene: conservation |E(t)-E(0)|/(K0+|U0|) and truth-similarity |E_m(t)-E_t4(t)|/(K0+|U0|) at ticks 100/1000).
Usage: PYTHONPATH=. python scripts/scatter_lossenergy_summarize.py <losstest dir> <cons root with one subdir per arm>"""
import glob
import json
import os
import sys

import numpy as np


def losstest(d):
    for f in sorted(glob.glob(f"{d}/*_4000.json")):
        if os.path.basename(f).startswith("regress_"):
            continue
        r = json.load(open(f))
        print(os.path.basename(f)[:-5], "calib", r.get("calib"))
        for fam in ("ab", "cl"):
            rows = [v for k, v in r["eval"].items() if k.startswith(fam)]
            err = np.mean([x["err"] for x in rows], 0)
            e = {k: np.mean([x["e20"][k] for x in rows]) for k in ("median", "mean", "p99")}
            es = {t: (np.mean([x["esim"][t]["abs_mean"] for x in rows]), np.mean([x["esim"][t]["signed_mean"] for x in rows])) for t in ("t20", "t100")}
            print(f"  {fam} err@5/10/20/50/100 {np.round(err, 4).tolist()} e20 med/mean/p99 {e['median']:.4f}/{e['mean']:.4f}/{e['p99']:.4f} "
                  f"esim@20 {es['t20'][0]:.4g} ({es['t20'][1]:+.3g}) esim@100 {es['t100'][0]:.4g} ({es['t100'][1]:+.3g}) dE100(cons) {np.mean([x['dE100'] for x in rows]):.4g}")


def cons(root):
    for arm in sorted(os.listdir(root)):
        for reg in ("nr100", "bh300"):
            f = f"{root}/{arm}/{reg}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            ticks = list(z["ticks"])
            sc = z["m_K"][0] + np.abs(z["m_U"][0])
            line = []
            for t in (100, 1000):
                i = ticks.index(t)
                c = np.abs(z["m_E"][i] - z["m_E"][0]) / sc
                s = (z["m_E"][i] - z["t4_E"][i]) / sc
                ct = np.abs(z["t4_E"][i] - z["t4_E"][0]) / sc
                line.append(f"@{t} cons dE med {np.median(c):.4g} per-seed {np.round(c, 3).tolist()} | sim |Em-Et| med {np.median(np.abs(s)):.4g} signed {np.round(s, 3).tolist()} | truth dE {np.median(ct):.2g}")
            print(arm, reg, " || ".join(line))


if __name__ == "__main__":
    losstest(sys.argv[1])
    if len(sys.argv) > 2:
        cons(sys.argv[2])
