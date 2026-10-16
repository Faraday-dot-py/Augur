"""Summarize lr-warmup arms from results/lrwarm (docs/debugging/lr-warmup.md)."""
import glob
import json
import sys

import numpy as np

d = sys.argv[1]
for a in ("ctl", "w1", "w2"):
    f = json.load(open(f"{d}/lrw_{a}/lrwarm_{a}_s4738.json"))["force"]
    print(a, "nn<1 force % by step:", {k: round(100 * v["nn<1"], 2) for k, v in f.items()})
for tag, fn in (("fresh 9500/9400", "iter_fresh_{}.json"), ("9300/9200", "iter_{}.json")):
    for a in ("frozenE", "ctl", "w1", "w2"):
        p = f"{d}/lrw_eval/" + fn.format(a)
        try:
            r = json.load(open(p))
        except FileNotFoundError:
            continue
        for ck, v in r.items():
            if a != "frozenE" and not ck.endswith("E_ms_kp_pot_v_g128.pt"):
                continue
            print(tag, a, "expB force %.2f nn<1 %.2f | @20 %.4f @100 %.3f | cluster force %.2f @20 %.4f @100 %.3f" % (
                100 * v["expB"]["force"]["all"]["rel_rms"], 100 * v["expB"]["force"]["nn 0.0-0.25"]["rel_rms"], v["expB"]["rollout"]["20"], v["expB"]["rollout"]["100"],
                100 * v["cluster"]["force"]["all"]["rel_rms"], v["cluster"]["rollout"]["20"], v["cluster"]["rollout"]["100"]))
for a in ("ctl", "w1", "w2"):
    out = {}
    for r in (1, 2, 3):
        for fp in glob.glob(f"{d}/lrw_cons/{a}_r{r}/*.json"):
            out.setdefault("files", []).append(fp)
    print(a, "cons files", sorted(out.get("files", []))[:6])
