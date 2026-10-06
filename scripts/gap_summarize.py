import json
import sys

for path in sys.argv[1:]:
    r = json.load(open(path))
    for k, v in r.items():
        m = v["9000"]
        e = m["err"]
        bd = m["breakdown"]["err2_at20_by_min_nn"]
        tf = m["teacher_forced"]
        print(k, "it", v["train_iters"], f"{v['train_seconds']:.0f}s", "err@5/10/20/50/100", [round(e[i], 5) for i in (4, 9, 19, 49, 99)])
        print("   err2@20 share by min-nn", {b: (round(x["share"], 2), round(x["n_frac"], 2)) for b, x in bd.items()})
        print("   scene err20 p10/50/90", [round(x, 4) for x in m["breakdown"]["scene_err20_pctl_10_50_90"]], "top10% share", round(m["breakdown"]["scene_top10pct_share_sq_err20"], 2))
        print("   tf rel", round(tf["overall_rel_rms"], 4), {b: round(x["rel_rms"], 4) for b, x in tf["bins"].items()}, "dE@100", round(m["dE"], 4))
        lc = v["loss_curve"]
        print("   loss", [(i, f"{l:.1e}") for i, l in lc[:: max(1, len(lc) // 6)]])
