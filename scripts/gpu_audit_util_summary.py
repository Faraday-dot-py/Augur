import csv
import glob
import json
import os
import sys


def pct(x, q):
    x = sorted(x)
    return x[min(len(x) - 1, int(q / 100 * len(x)))]


def avg(x):
    return sum(x) / len(x)


d = sys.argv[1]
num = lambda s: float(s.strip().split()[0])
for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
    rows = [r for r in list(csv.reader(open(f)))[1:] if len(r) >= 7]
    u = [num(r[1]) for r in rows]
    m = [num(r[2]) for r in rows]
    mem = [num(r[3]) for r in rows]
    tot = num(rows[0][4])
    pw = [num(r[5]) for r in rows]
    print(f"{os.path.basename(f):18s} n={len(u):4d} util avg {avg(u):5.1f} p95 {pct(u, 95):5.1f} | memutil avg {avg(m):5.1f} p95 {pct(m, 95):5.1f} | mem max {max(mem):7.0f}/{tot:.0f} MiB | power avg {avg(pw):5.0f} max {max(pw):5.0f} W")
print(json.dumps(json.load(open(os.path.join(d, "profile.json"))), indent=1))
