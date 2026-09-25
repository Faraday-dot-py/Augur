import sys

import numpy as np

d = np.load(sys.argv[1])
t, m, ev = d["truth"], d["model"], int(d["every"])
for f in [0, 5, 25, 50, 100, 250, 500]:
    def st(a):
        c = np.median(a[f], 0)
        r = np.linalg.norm(a[f] - c, axis=1)
        return np.percentile(r, [50, 90, 99]).round(1), np.linalg.norm(a[f].mean(0) - a[0].mean(0)).round(2)
    gap = np.linalg.norm(m[f] - t[f], axis=1).mean()
    print(f"step {f * ev}: gap {gap:.1f} | truth r50/90/99 {st(t)[0]} | model r50/90/99 {st(m)[0]} COM drift {st(m)[1]}")
