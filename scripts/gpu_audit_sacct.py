import re
import sys
from collections import defaultdict

TYPES = [
    ("rendering/video", r"video|render|lens|panels|gdir-video|snaps"),
    ("BH/gravity scaling", r"bh|orbit|scaling|scal-|bench|binary|clumpy|flyby|gravity|far|dual|tiling|tile|timing|1m|solar|mesh|lut|adaptive|granularity"),
    ("eval/regress/conservation", r"eval|regress|ood|cons-|cons1|conserv|errgrowth|rootcause|tails|truthchk|diag|select|probe|floors|chaos|harness|testaudit|audit|check|energy-probe|verify|interp|genker|genic|delete-star|threshold|sweep|compare"),
    ("training", r"train|stage|flownet|token-model|lt-main|d2b|d3b|d3-|ftde|softcap|windowed|cl-|cl$|pairfull|lj-|sf-|a3|gap-|joint|pilot|ft-|fix|fallback|contact|s46|d5|d4|d6|d1|recabl|int-|rollout|ball|pp8|bfrozen|energy|cons|opt-|shift|oracle|causal"),
    ("smoke/test", r"smoke|test|dbg|mcp"),
]


def hours(s):
    d = 0
    if "-" in s:
        d, s = s.split("-")
    p = s.split(":")
    h, m, sec = p if len(p) == 3 else (0, p[0], p[1])
    return int(d) * 24 + int(h) + int(m) / 60 + float(sec) / 3600


def classify(name):
    if re.search(r"smoke|dbg|mcp_|gpu_test", name):
        return "smoke/test"
    for label, pat in TYPES:
        if re.search(pat, name):
            return label
    return "other"


jobs = {}
for line in open(sys.argv[2]):
    f = line.rstrip("\n").split("|")
    if len(f) >= 6 and f[0].isdigit() and "gres/gpu" in f[3]:
        jobs[f[0]] = (f[1], hours(f[4]), f[5])
cpu = {}
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("|")
    if len(f) < 7 or not f[0][:1].isdigit():
        continue
    jid = f[0]
    if "." not in jid:
        continue
    elif jid.endswith(".batch"):
        m = re.search(r"cpu=([\d:\-]+)", f[5])
        if m:
            cpu[jid[:-6]] = hours(m.group(1))

agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for jid, (name, el, st) in jobs.items():
    if st.startswith(("RUNNING", "PENDING")):
        continue
    t = classify(name)
    a = agg[t]
    a[0] += 1
    a[1] += el
    if jid in cpu and el > 0.02:
        a[2] += cpu[jid]
        a[3] += el
print(f"{'type':28s} {'n':>4s} {'gpu_h':>7s} {'cpu/elapsed':>11s}")
tot = [0, 0.0]
for t, a in sorted(agg.items(), key=lambda x: -x[1][1]):
    r = a[2] / a[3] if a[3] else float("nan")
    print(f"{t:28s} {a[0]:4d} {a[1]:7.2f} {r:11.2f}")
    tot[0] += a[0]
    tot[1] += a[1]
print("total", tot)
for jid, (name, el, st) in sorted(jobs.items(), key=lambda x: -x[1][1])[:15]:
    print(jid, name, round(el, 2), st, round(cpu.get(jid, 0) / el, 2) if el else "")
