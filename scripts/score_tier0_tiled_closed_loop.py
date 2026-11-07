"""Scores a tiled closed-loop tier-0 run against the pre-registered lines (docs/debugging/experiment-log.md, 2026-10-26 tiled closed-loop tier 0):
PASS iff, on every seed 9100/9200/9300, tiled exact-field err@20 <= 1.25x the tiled baseline and dE@100 <= 2.5x the baseline, AND mean plain err@20
<= 1.10x the matched-60k tier 0, AND both mid-density betas (bins 2 and 3, 8-16 and 16-32 from the window edge) within +/-0.15.
Usage: python scripts/score_tier0_tiled_closed_loop.py <eval json> <hext json> [--out file.json]"""
import json
import sys

BASE = "results/loss_term_arms"
SEEDS = ("9100", "9200", "9300")


def beta_mid(hext):
    out = []
    for b in (2, 3):
        g = [r for r in hext["rows"] if r["scene"] == "mid" and r["mode"] == "ext" and r["bin"] == b]
        out.append(sum(r["numc"] for r in g) / max(sum(r["denc"] for r in g), 1e-12))
    return out


def plain20(ev):
    v = [r["err5_10_20"][2] for k, r in ev.items() if k.startswith("plain_")]
    return sum(v) / len(v)


def score(ev, hx):
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))["BL"]
    bp = json.load(open(f"{BASE}/baseline_plain_eval_matched60k.json"))["res"]
    out = {"err20_over_BL": {s: ev["tiled"][s]["err@20"] / bh[s]["err@20"] for s in SEEDS},
           "dE100_over_BL": {s: ev["tiled"][s]["dE_model"] / bh[s]["dE_model"] for s in SEEDS},
           "plain_err20": plain20(ev["plain"]), "matched60k_plain_err20": plain20(bp)}
    out["plain_over_matched60k"] = out["plain_err20"] / out["matched60k_plain_err20"]
    out["beta_mid"] = beta_mid(hx) if hx else None
    out["force_rel_free_rollout"] = {s: ev["tiled"][s]["force_rel_free_rollout"] for s in SEEDS}
    out["checks"] = {"err20": all(v <= 1.25 for v in out["err20_over_BL"].values()), "dE100": all(v <= 2.5 for v in out["dE100_over_BL"].values()),
                     "plain": out["plain_over_matched60k"] <= 1.10, "beta_mid": bool(out["beta_mid"]) and all(abs(b) <= 0.15 for b in out["beta_mid"])}
    out["PASS"] = all(out["checks"].values())
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out:
        a.remove(out)
    r = score(json.load(open(a[0])), json.load(open(a[1])) if len(a) > 1 else None)
    print(json.dumps(r, indent=1))
    if out:
        json.dump(r, open(out, "w"), indent=1)
