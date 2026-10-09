import json
import numpy as np

lines = [l for l in open("/home/awebb/polaris-mcp-files/square_wall_probe_3444.log") if l.startswith("ROLL ")]
r = json.loads(lines[0][5:])
for nm in ("model", "truth"):
    R = np.array(r[nm + "_r"])
    mx = R.max(0)
    print(nm, "max r pct 50/90/99/100", np.percentile(mx, [50, 90, 99, 100]).round(1), "n tracers max r>70:", int((mx > 70).sum()), "of", R.shape[1], "final r>70:", int((R[-1] > 70).sum()))
