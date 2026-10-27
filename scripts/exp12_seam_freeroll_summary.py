"""Summarise results/seam_freeroll/seam_all.json against the pre-registered rule (docs/debugging/experiment-log.md, free-rollout seam diagnosis)."""
import json
import sys


r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/seam_freeroll/seam_all.json"))
for arm, seeds in r.items():
    for sd, v in seeds.items():
        tot_w = sum(x["sum_w"] for k in ("kept", "changed") for x in v[k].values())
        n_all = sum(x["n"] for k in ("kept", "changed") for x in v[k].values())
        nc = sum(x["n"] for x in v["changed"].values())
        wc = sum(x["sum_w"] for x in v["changed"].values())
        band = ["seam<1", "seam<2", "seam<4"]
        nb = sum(v[k][b]["n"] for k in ("kept", "changed") for b in band if b in v[k])
        wb = sum(v[k][b]["sum_w"] for k in ("kept", "changed") for b in band if b in v[k])
        ratios = []
        for b, xc in v["changed"].items():
            xk = v["kept"].get(b)
            if xk and xc["n"] > 30:
                ratios.append((b, xc["w"] / xk["w"] if xk["w"] else float("nan"), xc["abs_w"] / xk["abs_w"], xc["rel_force_err"] / xk["rel_force_err"]))
        cw = v["kept"]
        cross = {b: round(x["cross_win_rel"], 3) for b, x in cw.items()}
        print(f"{arm} {sd}: total_w {tot_w:.1f}  changed n={nc} ({100 * nc / n_all:.2f}%) sum_w {wc:.1f} ({100 * wc / tot_w:.1f}%)  seam<4 pop {100 * nb / n_all:.1f}% sum_w share {100 * wb / tot_w:.1f}%")
        print("   changed/kept per bin (mean w, mean|w|, rel force err):", [(b, round(a, 2), round(c, 2), round(d, 2)) for b, a, c, d in ratios])
        print("   cross-window rel disc (kept):", cross)
        if "field_overlap" in v and v["field_overlap"]:
            print("   field overlap rel diff:", {t: {k: round(x, 3) for k, x in d.items()} for t, d in v["field_overlap"].items()})
