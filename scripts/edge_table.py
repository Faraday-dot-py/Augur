"""Parse EVAL/PROBE lines of edge_eval logs into one table (mean over held-out scene seeds 9000/12000). Usage: python3 scripts/edge_table.py log1 log2 ..."""
import re
import sys
from collections import defaultdict

rows = defaultdict(lambda: defaultdict(list))
for path in sys.argv[1:]:
    for line in open(path):
        m = re.match(r"EVAL (\S+) \[(\S+)\] err@5/10/20 (.*)", line)
        if m:
            key = (m.group(1), m.group(2))
            for kind in ("expB", "cluster"):
                v = [tuple(map(float, x.split("/"))) for x in re.findall(rf"{kind}_\d+:([\d.e+-]+/[\d.e+-]+/[\d.e+-]+)", m.group(3))]
                rows[key][kind] = [sum(t[i] for t in v) / len(v) for i in range(3)]
            continue
        m = re.match(r"PROBE (\S+) \[(\S+)\] static seed \d+ ratio ([\d.]+) signmatch ([\d.]+) changes ([\d.]+)", line)
        if m:
            key = (m.group(1), m.group(2))
            rows[key]["ratio"].append(float(m.group(3)))
            rows[key]["sign"].append(float(m.group(4)))
            continue
        m = re.match(r"PROBE (\S+) \[(\S+)\] roll seed \d+ beyond70/400 ([\d.]+) rev ([\d.]+)", line)
        if m:
            key = (m.group(1), m.group(2))
            rows[key]["b70"].append(float(m.group(3)))
            rows[key]["rev"].append(float(m.group(4)))
mean = lambda x: sum(x) / len(x) if x else float("nan")
print(f"{'model':14s} {'config':12s} {'band ratio':>10s} {'sign':>5s} {'b70/400':>8s} {'rev%':>6s} | expB err@5/10/20 | cluster err@5/10/20")
for (mdl, cfg), r in rows.items():
    print(f"{mdl:14s} {cfg:12s} {mean(r['ratio']):10.1f} {mean(r['sign']):5.2f} {mean(r['b70']):8.1f} {100 * mean(r['rev']):6.2f} | "
          + "/".join(f"{x:.4f}" for x in r["expB"]) + " | " + "/".join(f"{x:.4f}" for x in r["cluster"]))
