"""Scores visited-state-label-replay-tier-0 against its pre-registered lines (docs/theory/tiered-research-directions-2026-10-29.md, item 3). Per seed 9100/9200/9300:
free-rollout force error <= 1.25x tiled baseline, dE@100 <= 2.5x baseline, tiled err@20 <= 1.0x baseline; mean plain err@20 <= 1.10x the e44b796 checkpoint's.
KILL: dE@100 > 3.5x baseline on the mean over seeds or tiled err@20 > 1.25x baseline on any seed.
Usage: python scripts/score_tier0_visited_state_replay.py <new eval json> <e44b796 eval json> --out file.json"""
import json
import sys

from scripts.score_tier0_tiled_closed_loop import BASE, SEEDS, plain20


def score(ev, ref):
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))["BL"]
    r = lambda e, key, s: e["tiled"][s][key]
    out = {"force_free_rollout_over_BL": {s: r(ev, "force_rel_free_rollout", s) / bh[s]["force_rel"] for s in SEEDS},
           "dE100_over_BL": {s: r(ev, "dE_model", s) / bh[s]["dE_model"] for s in SEEDS},
           "err20_over_BL": {s: r(ev, "err@20", s) / bh[s]["err@20"] for s in SEEDS},
           "e44b796_dE100_over_BL": {s: r(ref, "dE_model", s) / bh[s]["dE_model"] for s in SEEDS},
           "e44b796_err20_over_BL": {s: r(ref, "err@20", s) / bh[s]["err@20"] for s in SEEDS},
           "plain_over_e44b796": plain20(ev["plain"]) / plain20(ref["plain"])}
    out["checks"] = {"force": all(v <= 1.25 for v in out["force_free_rollout_over_BL"].values()), "dE100": all(v <= 2.5 for v in out["dE100_over_BL"].values()),
                     "err20": all(v <= 1.0 for v in out["err20_over_BL"].values()), "plain": out["plain_over_e44b796"] <= 1.10}
    out["PASS"] = all(out["checks"].values())
    out["KILL"] = sum(out["dE100_over_BL"].values()) / 3 > 3.5 or any(v > 1.25 for v in out["err20_over_BL"].values())
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    o = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if o:
        a.remove(o)
    res = score(json.load(open(a[0])), json.load(open(a[1])))
    print(json.dumps(res, indent=1))
    if o:
        json.dump(res, open(o, "w"), indent=1)
