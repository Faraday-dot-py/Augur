import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.loads([l for l in open(sys.argv[1]) if l.startswith("RES ")][0][4:])
xs = np.array(d["xs"]); ex = np.array(d["exact"])
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
summ = {}
for a, nm in zip(ax, ("g128_e64", "g256_e128", "g128_e64_shift10")):
    r = d[nm]
    sh = 10.0 if "shift" in nm else 0.0
    x = xs
    a.plot(x, ex, "k", label="exact (unshifted scene)")
    for k, c in (("kernel_only", "g"), ("net_only", "m"), ("total", "r")):
        a.plot(x, r[k], c, label=k, lw=1)
    a.set_ylim(-3000, 3000); a.set_title(nm + (" (scene+tracers shifted +10; edge at x=22)" if sh else "")); a.legend(fontsize=7)
    edge = 32.0 if "e64" in nm else 64.0
    edge = edge - sh
    a.axvline(edge, c="gray")
    tot = np.array(r["total"]); ker = np.array(r["kernel_only"]); net = np.array(r["net_only"])
    m = (x > edge - 8) & (x < edge + 2)
    summ[nm] = {"edge_in_x": edge, "max|total|_within_[edge-8,edge+2]": float(np.abs(tot[m]).max()), "max|kernel|_same": float(np.abs(ker[m]).max()), "max|net|_same": float(np.abs(net[m]).max()),
                "max|exact|_same": float(np.abs(ex[m]).max()),
                "total_at_edge-3..edge+1": [(float(a_), round(float(b), 0)) for a_, b in zip(x[(x >= edge - 3) & (x <= edge + 1)], tot[(x >= edge - 3) & (x <= edge + 1)])]}
plt.tight_layout(); plt.savefig("results/square_wall/square_wall_probe2.png", dpi=65)
json.dump(summ, open("results/square_wall/square_wall_probe2_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
