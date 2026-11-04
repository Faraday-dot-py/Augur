"""Scores the Muon-on-flattened-conv-kernels run from the tiled lines of an eval log when the eval's final plain stage ran out of GPU memory (11 GiB sandbox card) and no eval json was written.
The PASS/KILL gate uses only the tiled blocks, so the verdict is the one scripts/score_tiered_muon_conv_flattened.py would give; the plain ratio is not computed.
Usage: PYTHONPATH=. python scripts/score_tiered_muon_conv_flattened_from_log.py <eval log> --eval-out <tiled-only eval json> --out <score json>"""
import json
import sys

from scripts.score_tiered_muon_conv_flattened import FROZEN, ratios
from scripts.score_tier0_tiled_closed_loop import BASE, SEEDS


def parse(path):
    ev = {"tiled": {}, "tiled_no_exchange": {}}
    for line in open(path):
        p = line.split(" ", 2)
        if len(p) == 3 and p[0] in ev and p[1] in SEEDS:
            ev[p[0]][p[1]] = json.loads(p[2])
    return ev


if __name__ == "__main__":
    a = sys.argv
    ev = parse(a[1])
    assert all(s in ev["tiled"] for s in SEEDS), ev["tiled"].keys()
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))["BL"]
    fz = json.load(open(FROZEN))
    err, dE = ratios(ev, bh)
    ferr, fdE = ratios(fz, bh)
    out = {"err20_over_BL": err, "dE100_over_BL": dE, "plain_over_matched60k": None, "frozen_4060_err20_over_BL": ferr, "frozen_4060_dE100_over_BL": fdE,
           "force_rel_free_rollout": {s: ev["tiled"][s]["force_rel_free_rollout"] for s in SEEDS}}
    out["PASS"] = all(v <= 0.82 for v in err.values()) and all(v <= 2.5 for v in dE.values())
    out["verdict"] = "PASS" if out["PASS"] else "KILL"
    print(json.dumps(out, indent=1))
    json.dump(ev, open(a[a.index("--eval-out") + 1], "w"), indent=1)
    json.dump(out, open(a[a.index("--out") + 1], "w"), indent=1)
