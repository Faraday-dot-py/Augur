"""Print the scene x tier summary table from results/scenes/*.json."""
import json
import os

R = "results/scenes"


def load(name):
    p = f"{R}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=3):
    return "-" if x is None else f"{x:.{n}g}"


print("bounce (err@5/10/20/100, outside-box@100, restitution@v=10/15)")
for tag in ("zeroshot", "ft1", "ft4", "full4"):
    d = load("bounce_" + tag)
    if d:
        m = d["mean"]["err"]
        r = d["restitution"]
        print(tag, [f(m[k]) for k in ("5", "10", "20", "100")], f(d["mean"]["outside_box_frac"]["100"]), [f((r.get(v) or {}).get("model")) for v in ("10", "15")])
d = load("bounce_zeroshot")
print("ballistic", [f(d["mean"]["ballistic_gravity"][k]) for k in ("5", "10", "20", "100")], "floor", [f(d["mean"]["floor_pert1e-5"][k]) for k in ("5", "10", "20", "100")])
print("orbit (err@20/100/1000, period err med, lost, dE/E@1000, dL/L@1000)")
for fam in ("binary", "planetary"):
    for tag in ("zeroshot", "ft", "ft2", "full"):
        d = load("orbit_" + tag)
        if d and fam in d:
            m = d[fam]["mean"]
            print(fam, tag, [f(m["err"][k]) for k in ("20", "100", "1000")], f(m["period_err_median"]), f(m["planets_lost_frac"]), f(m["dE_over_E_model_at"]["1000"]), f(m["dL_over_L_model_at"]["1000"]))
print("globular N=300 (err@20, r50 band, E band, dE/E end, truth dE/E)")
for tag in ("zeroshot", "ft"):
    d = load("globular_" + tag)
    if d:
        for n in ("100", "300", "1000"):
            runs = d["runs"][n]
            av = lambda k: sum(r[k] for r in runs) / len(runs)
            print(tag, n, f(sum(r["err"]["20"] for r in runs) / len(runs)), f(av("r50_band_frac")), f(av("e_band_frac")), f(av("dE_over_E_model_end")), f(av("dE_over_E_truth_end")), "esc", f(av("esc_model_end")), f(av("esc_truth_end")))
print("bh")
d = load("bh_zeroshot")
for k, v in d["variants"].items():
    print(k, {a: f(b) for a, b in v["mean_err"].items()}, "floor@50", f(v["mean_floor"]["50"]))
