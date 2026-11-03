"""One line per scene: held-out (seeds 9000, 12000) hole-4 beta pooled over ticks 5/10/20 per edge bin, from hetero_a_<tag>_<scene>.json.
Usage: python scripts/hetero_beta_table.py <dir> <tag>"""
import json
import sys

BINS = ("0-4", "4-8", "8-16", "16-32")
HELD = tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "9000,12000").split(","))
d, tag = sys.argv[1], sys.argv[2]
for sc in ("sparse", "mid", "dense"):
    try:
        rows = json.load(open(f"{d}/hetero_a_{tag}_{sc}.json"))["rows"]
    except FileNotFoundError:
        continue
    rs = [r for r in rows if "kind" not in r and r["seed"] in HELD and r["tick"] in (5, 10, 20)]
    out = []
    for b in range(4):
        g = [r for r in rs if r["bin"] == b]
        num, den = sum(r["num"] for r in g), sum(r["den"] for r in g)
        out.append(f"{BINS[b]} {num / max(den, 1e-12):+.3f}")
    print(f"BETA {tag} {sc}: " + " | ".join(out), flush=True)
