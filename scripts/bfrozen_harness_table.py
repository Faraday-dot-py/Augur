"""Key scatter_regress rows for the B-frozen comparison. Usage: python3 scripts/bfrozen_harness_table.py DIR OUT.md"""
import json
import sys

import numpy as np

TAGS = [("E", "E (eager)"), ("Bf1", "Bf1"), ("Bf2", "Bf2"), ("B2", "B2 (free-space)"), ("C", "C (free-space)"), ("Cfz", "frozen-C (free-space)")]


def main():
    d, out = sys.argv[1], sys.argv[2]
    L = ["| model | nr100 dE@1000 | nr100 dL | bh300 dE@1000 | bh300 dL | two_body err@5/@20 | expB err@5/@20 | bh300 err@5/@20 |", "|---|---|---|---|---|---|---|---|"]
    for t, name in TAGS:
        r = json.load(open(f"{d}/h_{t}.json"))
        c = r["conservation"]
        cell = lambda reg: " / ".join(f"{np.mean([r['regimes'][reg]['per_seed'][s]['err'][k] for s in r['regimes'][reg]['per_seed']]):.3g}" for k in ("5", "20"))
        L.append(f"| {name} | {c['nr100']['model']['dE']:.4g} | {c['nr100']['model']['dL']:.3g} | {c['bh300']['model']['dE']:.4g} | {c['bh300']['model']['dL']:.3g} | {cell('two_body')} | {cell('expB')} | {cell('bh300')} |")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
