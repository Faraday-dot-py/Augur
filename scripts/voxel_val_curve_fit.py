import re
import sys

import numpy as np
from scipy.optimize import curve_fit

FIRST_ITER_FIT = int(sys.argv[2]) if len(sys.argv) > 2 else 14400
RATIO_AT_10K = 7.7749
VAL_AT_10K = 0.13518
KILL_RATIO = 4.0
PASS_RATIO = 2.0


def load(path):
    pts = {}
    for m in re.finditer(r"val err@20 at it (\d+): ([0-9.]+)", open(path).read()):
        pts[int(m.group(1))] = float(m.group(2))
    return np.array(sorted(pts.items()))


def power(x, a, b, c):
    return a * (x / 1e4) ** (-b) + c


def power_nofloor(x, a, b):
    return a * (x / 1e4) ** (-b)


d = load(sys.argv[1])
fit = d[d[:, 0] >= FIRST_ITER_FIT]
print("points used", len(fit), "of", len(d), "from it", FIRST_ITER_FIT)
for name, f, p0 in (("floor", power, (0.3, 0.5, 0.05)), ("no-floor", power_nofloor, (0.3, 0.5))):
    try:
        p, cov = curve_fit(f, fit[:, 0], fit[:, 1], p0=p0, maxfev=20000)
    except Exception as e:
        print(name, "fit failed", e)
        continue
    res = fit[:, 1] - f(fit[:, 0], *p)
    print(name, "params", np.round(p, 4), "rmse", round(float(np.sqrt((res ** 2).mean())), 4))
    for it in (30000, 45000, 60000, 120000):
        print("  pred val err@20 at", it, round(float(f(it, *p)), 4))
    for label, ratio in (("kill line ratio 4.0", KILL_RATIO), ("pass line ratio 2.0", PASS_RATIO)):
        target = VAL_AT_10K * ratio / RATIO_AT_10K
        xs = np.arange(FIRST_ITER_FIT, 2_000_001, 500)
        hit = xs[f(xs, *p) <= target]
        print("  ", label, "needs val", round(target, 4), "->", int(hit[0]) if len(hit) else "never (floor above target)")
