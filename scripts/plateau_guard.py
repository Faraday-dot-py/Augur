"""Plateau guard (user decision 2026-11-09, item 1). Reads <ckpt>.val.jsonl written by train_scatter_field.py --val-every.
UNDERTRAINED if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window.
Usage: python scripts/plateau_guard.py VAL.jsonl TOTAL_ITERS"""
import json
import sys

import numpy as np

FRAC = 0.10
DROP = 0.03


def guard(rows, total):
    last = {}
    for r in rows:
        last[r["it"]] = r["val_loss"]
    lo2, lo1 = total * (1 - 2 * FRAC), total * (1 - FRAC)
    a = [v for it, v in last.items() if lo2 <= it < lo1]
    b = [v for it, v in last.items() if it >= lo1]
    if len(a) < 3 or len(b) < 3:
        return {"status": "NO-DATA", "n_prev": len(a), "n_last": len(b)}
    ma, mb = float(np.mean(a)), float(np.mean(b))
    drop = (ma - mb) / ma
    return {"status": "UNDERTRAINED" if drop > DROP else "PLATEAUED", "mean_prev": ma, "mean_last": mb, "drop": drop, "n_prev": len(a), "n_last": len(b)}


def main():
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    res = guard(rows, int(sys.argv[2]))
    print(json.dumps(res))
    sys.exit(0 if res["status"] in ("PLATEAUED", "UNDERTRAINED") else 3)


if __name__ == "__main__":
    main()
