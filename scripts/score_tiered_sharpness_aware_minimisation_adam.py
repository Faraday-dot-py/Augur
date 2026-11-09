"""Scores sharpness-aware minimisation around Adam (rho 0.05) on the tiled closed-loop tier 0 against the same line as the Muon swap (docs/debugging/optimizer-sweep-proposal.md section 4.3):
PASS iff on every seed 9100/9200/9300 dE@100 <= 2.5x the tiled baseline (BL) AND tiled err@20 <= 0.82x BL; KILL otherwise. No argmin on eval seeds: the final checkpoint is scored.
Also reports the frozen job 4060 numbers from its eval json for comparison (not gated); 4060 itself is 0.8215x at seed 9100, above the 0.82 err line.
Usage: python scripts/score_tiered_sharpness_aware_minimisation_adam.py <eval json> [--out file.json]"""
import json
import sys

from scripts.score_tier0_tiled_closed_loop import BASE, SEEDS, plain20

FROZEN = "results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo_eval.json"


def ratios(ev, bh):
    return ({s: ev["tiled"][s]["err@20"] / bh[s]["err@20"] for s in SEEDS}, {s: ev["tiled"][s]["dE_model"] / bh[s]["dE_model"] for s in SEEDS})


def score(ev):
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))["BL"]
    bp = json.load(open(f"{BASE}/baseline_plain_eval_matched60k.json"))["res"]
    fz = json.load(open(FROZEN))
    err, dE = ratios(ev, bh)
    ferr, fdE = ratios(fz, bh)
    out = {"err20_over_BL": err, "dE100_over_BL": dE, "plain_over_matched60k": plain20(ev["plain"]) / plain20(bp),
           "frozen_4060_err20_over_BL": ferr, "frozen_4060_dE100_over_BL": fdE, "frozen_4060_plain_over_matched60k": plain20(fz["plain"]) / plain20(bp),
           "force_rel_free_rollout": {s: ev["tiled"][s]["force_rel_free_rollout"] for s in SEEDS}}
    out["PASS"] = all(v <= 0.82 for v in err.values()) and all(v <= 2.5 for v in dE.values())
    out["verdict"] = "PASS" if out["PASS"] else "KILL"
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
