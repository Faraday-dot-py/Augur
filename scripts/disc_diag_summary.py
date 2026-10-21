"""Print core/disc radii and escape fraction every --every ticks from results/disc/<tag>/diag.json."""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--every", type=int, default=200)
a = ap.parse_args()
d = json.load(open(f"{a.dir}/diag.json"))
print(a.dir.split("/")[-1], [(x["tick"], round(x["core_in2"], 3), round(x["r50"], 2), round(x["r90"], 2), round(x["frac_outside"], 3)) for x in d if x["tick"] % a.every == 0])
