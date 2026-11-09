"""Ratios against the tiled baseline BL's own error (no pass/kill) and the pre-registered reduction fraction for each replay-mechanism arm:
fraction = (base dE@100/BL - arm dE@100/BL) / (base dE@100/BL - replay dE@100/BL), means over seeds 9100/9200/9300 (1.0 = reproduces the replay gain, 0 = no gain).
Usage: PYTHONPATH=. python scripts/summarise_replay_mechanism_arms.py --base <eval json> --replay <eval json> --out <json> arm-name=<eval json> ..."""
import argparse
import json

from scripts.score_tier0_tiled_closed_loop import BASE, SEEDS


def ratios(ev, bh):
    r = lambda key, s: ev["tiled"][s][key]
    return {"dE100_over_BL": [r("dE_model", s) / bh[s]["dE_model"] for s in SEEDS], "err20_over_BL": [r("err@20", s) / bh[s]["err@20"] for s in SEEDS],
            "force_free_rollout_over_BL": [r("force_rel_free_rollout", s) / bh[s]["force_rel"] for s in SEEDS]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("arms", nargs="*")
    a = ap.parse_args()
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))["BL"]
    mean = lambda x: sum(x) / len(x)
    res = {"base": ratios(json.load(open(a.base)), bh), "replay": ratios(json.load(open(a.replay)), bh), "arms": {}}
    b, r = mean(res["base"]["dE100_over_BL"]), mean(res["replay"]["dE100_over_BL"])
    for spec in a.arms:
        name, path = spec.split("=", 1)
        x = ratios(json.load(open(path)), bh)
        m = mean(x["dE100_over_BL"])
        x["mean_dE100_over_BL"] = m
        x["mean_err20_over_BL"] = mean(x["err20_over_BL"])
        x["reduction_fraction_of_replay_gain"] = (b - m) / (b - r)
        res["arms"][name] = x
    res["base"]["mean_dE100_over_BL"], res["replay"]["mean_dE100_over_BL"] = b, r
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: round(v["reduction_fraction_of_replay_gain"], 3) for k, v in res["arms"].items()}))


if __name__ == "__main__":
    main()
