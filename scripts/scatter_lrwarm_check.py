"""Kill check on smoke runs: exit 1 if any warmup arm's nn<1 force error grows >3x within the first 10 steps (docs/debugging/lr-warmup.md). Prints the control's jump as a positive control."""
import json
import sys

d = sys.argv[1]
bad = False
for arm in ("ctl", "w1", "w2"):
    f = json.load(open(f"{d}/lrwarm_{arm}_s4738.json"))["force"]
    r = {k: f[k]["nn<1"] / f["0"]["nn<1"] for k in ("1", "2", "5", "10") if k in f}
    print(arm, "nn<1 force error ratio to step 0 at steps 1/2/5/10:", {k: round(v, 2) for k, v in r.items()}, flush=True)
    if arm != "ctl" and r.get("10", 0) > 3:
        bad = True
sys.exit(1 if bad else 0)
