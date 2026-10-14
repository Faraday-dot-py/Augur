"""Summarize results/losstest/*.json from scatter_losstest.py: per arm, mean over seeds 9100/9200/9300 for ab and cl."""
import glob
import json
import os
import sys


class np:
    @staticmethod
    def mean(x, axis=None):
        x = list(x)
        if axis == 0:
            return [sum(c) / len(c) for c in zip(*x)]
        return sum(x) / len(x)

    @staticmethod
    def round(x, n):
        return type("A", (), {"tolist": lambda self: [round(v, n) for v in x]})()



d = sys.argv[1]
print("delta", json.load(open(f"{d}/delta.json")))
for f in sorted(glob.glob(f"{d}/*_4000.json")) + sorted(glob.glob(f"{d}/*evalonly.json")):
    r = json.load(open(f))
    for fam in ("ab", "cl"):
        rows = [v for k, v in r["eval"].items() if k.startswith(fam)]
        err = np.mean([x["err"] for x in rows], 0)
        e = {k: np.mean([x["e20"][k] for x in rows]) for k in ("median", "mean", "p90", "p99", "sq_loss", "top1pct_sq_share")}
        fo = {k: np.mean([x["force"][k] for x in rows]) for k in ("all", "<0.5", "0.5-1", ">=1")}
        print(os.path.basename(f)[:-5], fam, "err@5/10/20/50/100", np.round(err, 4).tolist(),
              "e20 med/mean/p90/p99 %.4f/%.4f/%.4f/%.4f sq %.2e top1 %.2f" % (e["median"], e["mean"], e["p90"], e["p99"], e["sq_loss"], e["top1pct_sq_share"]),
              "force all/<.5/.5-1/>=1 %.3f/%.3f/%.3f/%.3f" % tuple(fo.values()))
