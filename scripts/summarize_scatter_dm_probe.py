import json
import sys

from statistics import mean

res = json.load(open(sys.argv[1]))
arms = []
for r in res:
    if r["arm"] not in arms:
        arms.append(r["arm"])
keys = ["err_vis@5", "err_vis@10", "err_vis@20", "err_vis@50", "floor_vis@20", "ballistic_vis@20", "vc_relerr_median"]
print("arm | n | " + " | ".join(keys) + " | r50 model/truth | nonfinite")
for a in arms:
    rs = [r for r in res if r["arm"] == a]
    vals = [mean([r[k] for r in rs]) for k in keys]
    r50 = mean([r["r50_vis_end_model"] for r in rs]), mean([r["r50_vis_end_truth"] for r in rs])
    print(f"{a} | {rs[0]['n_total']} | " + " | ".join(f"{v:.3g}" for v in vals) + f" | {r50[0]:.2f}/{r50[1]:.2f} | {[r['nonfinite_tick'] for r in rs]}")
print("per-seed err_vis@20:")
for a in arms:
    print(a, [round(r["err_vis@20"], 3) for r in res if r["arm"] == a], "vc terciles", [[round(x, 2) for x in r["vc_relerr_terciles"]] for r in res if r["arm"] == a][:1])
