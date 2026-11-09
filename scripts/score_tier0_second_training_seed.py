"""Scores the second-training-seed tiled closed-loop tier 0 (matched-60k base from scratch at seed 4739, then the tiled closed-loop fine-tune) against the
lines pre-registered in docs/theory/supervisor-log.md (2026-11-03): PASS iff tiled err@20 <= 1.0x the tiled baseline on every seed 9100/9200/9300 AND mean plain
err@20 <= 1.10x the seed-4738 matched-60k value; KILL iff tiled err@20 > 1.25x the baseline on any seed; otherwise INCONCLUSIVE_PARTIAL. Drift spread (dE@100 ratio to
the baseline) is reported, not gated.
Usage: python scripts/score_tier0_second_training_seed.py <eval json> [--out file.json]"""
import json
import sys

from scripts.score_tier0_tiled_closed_loop import BASE, SEEDS, plain20


def score(ev):
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))["BL"]
    bp = json.load(open(f"{BASE}/baseline_plain_eval_matched60k.json"))["res"]
    err = {s: ev["tiled"][s]["err@20"] / bh[s]["err@20"] for s in SEEDS}
    dE = {s: ev["tiled"][s]["dE_model"] / bh[s]["dE_model"] for s in SEEDS}
    plain = plain20(ev["plain"]) / plain20(bp)
    out = {"err20_over_BL": err, "dE100_over_BL": dE, "dE100_over_BL_range": [min(dE.values()), max(dE.values())], "plain_over_matched60k": plain}
    out["pass_err20"] = all(v <= 1.0 for v in err.values())
    out["pass_plain"] = plain <= 1.10
    out["kill"] = any(v > 1.25 for v in err.values())
    out["verdict"] = "KILL" if out["kill"] else "PASS" if out["pass_err20"] and out["pass_plain"] else "INCONCLUSIVE_PARTIAL"
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out:
        a.remove(out)
    r = score(json.load(open(a[0])))
    print(json.dumps(r, indent=1))
    if out:
        json.dump(r, open(out, "w"), indent=1)
