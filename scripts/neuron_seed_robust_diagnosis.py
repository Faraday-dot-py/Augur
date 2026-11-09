import json
import os
import sys

import numpy as np

FILES = os.path.expanduser("~/polaris-mcp-files")
SLOW = {
    4738: "/home/awebb/Research/Augur/.claude/worktrees/timesfm-neuron-ideas/results/neuron_graph_slow_patch/seed4738/results.json",
    4739: f"{FILES}/seedrobust/slow-patch-seed4739.json",
    4740: f"{FILES}/seedrobust/slow-patch-seed4740.json",
}
PLAS = {
    4738: f"{FILES}/neuronplas/seed4738/results.json",
    4739: f"{FILES}/seedrobust/plasticity-seed4739.json",
    4740: f"{FILES}/seedrobust/plasticity-seed4740.json",
}
ARMS = ("baseline-full-synaptic-messages", "control-baseline-architecture-48-tick-segments-control", "variant-full-synaptic-messages")
P, FX = "plastic-synapse-state", "fixed-synapse-no-mutable-state"


def slow_table():
    rows = {}
    for seed, path in SLOW.items():
        d = json.load(open(path))
        for n in ("10", "100"):
            for arm in ARMS:
                r = d["eval"][n][arm]
                rs = r["rate_statistics"]["300"]
                rows[(seed, n, arm)] = dict(err20=r["err@20"][0], f1=r["spike_f1@20"][0], slope=rs["slope_pred_on_true"],
                                            corr=rs["correlation"], ratio=rs["ratio"])
        print(seed, "verdict", d["verdict"], "step-only", {k: {s: round(v, 3) for s, v in m.items()} for k, m in d.get("step_only_loss", {}).items()})
    print("seed N arm err20 f1@20 slope corr ratio")
    for (seed, n, arm), v in rows.items():
        print(seed, n, arm[:8], *(f"{v[k]:.3f}" for k in ("err20", "f1", "slope", "corr", "ratio")))
    for n in ("10", "100"):
        for key in ("err20", "f1", "slope"):
            b = np.array([rows[(s, n, ARMS[0])][key] for s in SLOW])
            c = np.array([rows[(s, n, ARMS[1])][key] for s in SLOW])
            v = np.array([rows[(s, n, ARMS[2])][key] for s in SLOW])
            print(f"N={n} {key}: baseline mean {b.mean():.3f} sd {b.std(ddof=1):.3f} | control mean {c.mean():.3f} sd {c.std(ddof=1):.3f} | variant mean {v.mean():.3f} sd {v.std(ddof=1):.3f}")
            print(f"   variant-baseline {np.round(v - b, 3)}  control-baseline {np.round(c - b, 3)}  variant-control {np.round(v - c, 3)}")
        for s in SLOW:
            b, v = rows[(s, n, ARMS[0])], rows[(s, n, ARMS[2])]
            print(f"   seed {s} err20 ratio variant/baseline {v['err20'] / b['err20']:.3f} (line 1.03)")


def stat(x):
    x = np.asarray(x)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def plas_table():
    for seed, path in PLAS.items():
        d = json.load(open(path))
        print("seed", seed, "verdict", d["verdict"])
        for n in ("10", "100"):
            for k in (0, 1):
                ep = d["episodes"][n]
                fwd = np.asarray(ep["forward"][P]["dR"])[:, k]
                rev = np.asarray(ep["reversed"][P]["dR"])[:, k]
                non = np.asarray(ep["none"][P]["dR"])[:, k]
                ffx = np.asarray(ep["forward"][FX]["dR"])[:, k]
                rfx = np.asarray(ep["reversed"][FX]["dR"])[:, k]
                tf = np.asarray(ep["forward"]["teacher"]["dR"])[:, k]
                cp = fwd.mean() - rev.mean()
                cf = ffx.mean() - rfx.mean()
                diff = fwd - rev
                m, se = stat(diff)
                print(f"  N={n} probe{k + 1}: plastic fwd {fwd.mean():+.2f} rev {rev.mean():+.2f} none {non.mean():+.2f} | fixed fwd {ffx.mean():+.2f} rev {rfx.mean():+.2f} | teacher fwd {tf.mean():+.2f} | contrast plastic {cp:+.2f} (paired SE {se:.2f}, t {m / se:+.2f}) fixed {cf:+.2f}")
        g = {n: {c: np.mean(d["episodes"][n][c][P]["g_final"]) for c in ("forward", "reversed", "none")} for n in ("10", "100")}
        print("  final g plastic", {n: {c: round(float(v), 2) for c, v in x.items()} for n, x in g.items()})


if __name__ == "__main__":
    slow_table()
    plas_table()
    sys.exit(0)
