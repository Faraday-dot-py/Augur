"""Print the diagnostic rows of galaxy runs from a Slurm log (JSON lines) or a diag.json. Usage: python3 scripts/summarize_galaxy.py <log or diag.json> [every]"""
import json
import sys

path = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rows = json.load(open(path)) if path.endswith(".json") else [json.loads(line) for line in open(path) if line.startswith('{"tick"')]
run = 0
print("tick  ke  pe  E  dE  virial  bound  r25  r50  r90  out  vmax/c  nonfinite")
last = -1
for d in rows:
    if d["tick"] <= last:
        run += 1
        print(f"--- run {run}")
    last = d["tick"]
    if d["tick"] % every == 0 or d["tick"] == rows[-1]["tick"]:
        print(f"{d['tick']:5d} {d['ke']:9.1f} {d['pe']:9.1f} {d['e']:9.1f} {d['dE']:+.4f} {d['virial']:.3f} {d['bound']:.4f} {d['r25']:.2f} {d['r50']:.2f} {d['r90']:.2f} {d['frac_outside']:.4f} {d['vmax_over_c']:.3f} {d['nonfinite']}")
