"""Scores frozen tiled tier-0 checkpoints on fresh held-out seeds (scripts/eval_tier0_fresh_seed_checkpoints.py) against the pre-registered line:
PASS iff tiled err@20 <= 1.0x the tiled baseline (BL, computed on the same scenes) on every fresh seed for EVERY checkpoint; KILL iff err@20 > 1.25x BL on any
(checkpoint, seed); otherwise INCONCLUSIVE. Final checkpoints, no selection on these seeds. dE@100/BL is reported, not gated. A seed whose BL err@20 is outside
0.5x-2x of the mean BL err@20 on seeds 9100/9200/9300 (results/loss_term_arms) is flagged as unrepresentative; the flag does not change the verdict.
Usage: python scripts/score_tier0_fresh_seed_checkpoints.py <eval json> [--out file.json]"""
import json
import sys

from scripts.score_tier0_tiled_closed_loop import BASE, SEEDS


def score(ev):
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))["BL"]
    ref = sum(bh[s]["err@20"] for s in SEEDS) / len(SEEDS)
    seeds = [str(s) for s in ev["seeds"]]
    names = [k for k in ev if k not in ("seeds", "BL")]
    out = {"err20_over_BL": {n: {s: ev[n][s]["err@20"] / ev["BL"][s]["err@20"] for s in seeds} for n in names},
           "dE100_over_BL": {n: {s: ev[n][s]["dE_model"] / ev["BL"][s]["dE_model"] for s in seeds} for n in names},
           "BL_err20": {s: ev["BL"][s]["err@20"] for s in seeds}, "BL_err20_9100_9200_9300_mean": ref,
           "unrepresentative_seeds": [s for s in seeds if not 0.5 * ref <= ev["BL"][s]["err@20"] <= 2 * ref]}
    flat = [v for n in names for v in out["err20_over_BL"][n].values()]
    out["PASS"] = all(v <= 1.0 for v in flat)
    out["KILL"] = any(v > 1.25 or v != v for v in flat)
    out["verdict"] = "PASS" if out["PASS"] else "KILL" if out["KILL"] else "INCONCLUSIVE"
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    o = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if o:
        a.remove(o)
    r = score(json.load(open(a[0])))
    print(json.dumps(r, indent=1))
    if o:
        json.dump(r, open(o, "w"), indent=1)
