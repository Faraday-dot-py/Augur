import json
import sys

import numpy as np


def main(path):
    r = json.load(open(path))
    print("verdict", r["verdict"], "params", r["parameters"], "wall", round(r["wall_seconds"]))
    for key, d in r["verdict_details"].items():
        print(key, {k: [round(x, 2) for x in v] if isinstance(v[0], float) else [[round(y, 2) for y in x] for x in v] for k, v in d.items() if k != "checks"})
        print("  failed:", [k for k, v in d["checks"].items() if not v])
    for n, conds in r["episodes"].items():
        for cond, arms in conds.items():
            line = []
            for a, row in arms.items():
                g = np.array(row["g_final"])
                line.append(f"{a[:12]} g_final {g.mean():.2f}+-{g.std():.2f}")
            print(f"N={n} {cond}: " + "; ".join(line))


if __name__ == "__main__":
    main(sys.argv[1])
