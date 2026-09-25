"""Prints markdown tables from results/lut_eval.json, lut_bench.json, lut_central.json (whichever exist)."""
import json
import os

import numpy as np


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


ev = load("results/lut_eval.json")
if ev:
    print("## table error (max abs / peak)")
    for s, t in ev["table_error"].items():
        print(s, {k: (round(v["max_abs"], 4), round(v["max_rel_of_peak"], 6)) for k, v in t.items()})
    print("## one-step max dv / max dp vs base")
    for st, d in ev["one_step"].items():
        print(st, "contact_frac", round(d["contact_frac"], 3), "median |dv|", round(d["base_dv_median"], 4))
        for s, r in d["variants"].items():
            print("  ", s, "max_dp %.2e max_dv %.2e mean_dv %.2e" % (r["max_dp"], r["max_dv"], r["mean_dv"]), "act", round(r.get("active_frac", -1), 3))
    print("## long rollouts")
    for sc, rows in ev["long"].items():
        print(sc, "truth E", rows["truth"]["E"], "KE", rows["truth"]["KE"], "contact", rows["truth"]["contact_frac"])
        for s, r in rows.items():
            if s == "truth":
                continue
            print("  ", s, "finite", r["finite"], "E100 %.1f E500 %.1f" % (r["E"]["100"], r["E"]["500"]), "KE100 %.1f KE500 %.1f" % (r["KE"]["100"], r["KE"]["500"]),
                  "cf", {k: round(v, 3) for k, v in r["contact_frac"].items()},
                  "div20 %.3g div100 %.3g div500 %.3g" % (r["div_vs_base"]["20"], r["div_vs_base"]["100"], r["div_vs_base"]["500"]),
                  "err_truth 5/20/100 %.3f %.3f %.1f" % (r["err_vs_truth"]["5"], r["err_vs_truth"]["20"], r["err_vs_truth"]["100"]), "act", round(r.get("active_frac_mean", -1), 3))
    print("## err@5/10/20 (mean) n20_b4 | n100_b100; E_model@100")
    for s, r in ev["eval"].items():
        a, b = r["n20_b4"], r["n100_b100"]
        print(s, "%.4f %.4f %.4f | %.4f %.4f %.4f" % tuple(a["err_mean"][k] for k in ("5", "10", "20")) + tuple(b["err_mean"][k] for k in ("5", "10", "20")) if False else
              s, [round(a["err_mean"][k], 4) for k in ("5", "10", "20")], [round(b["err_mean"][k], 4) for k in ("5", "10", "20")],
              "E100", round(a["E_model"]["100"], 2), round(b["E_model"]["100"], 1), "finite", a["finite_frac"], b["finite_frac"])
    print("## wall y ratio (speeds 3..80), x-floor dE, pair relKE offset 0")
    for s, r in ev["eval"].items():
        yw = [round(x["model_ratio"], 3) for x in r["wall"] if x["wall"] == "y_wall"]
        xw = [round(x["model_dE"], 1) for x in r["wall"] if x["wall"] == "x_floor"]
        pr = [round(x["model_relKE_ratio"], 3) for x in r["pair"] if x["offset"] == 0.0]
        pm = [round(x["model_pmom_err"], 2) for x in r["pair"] if x["offset"] == 0.0]
        print(s, "\n   yratio", yw, "\n   xdE", xw, "\n   relKE", pr, "\n   pmom", pm)

bn = load("results/lut_bench.json")
if bn:
    print("## bench ms/step")
    for row in bn:
        print(row["regime"], row["N"], "contact_frac %.3f" % row["contact_frac"], "edges", row["edges"], "vmax %.1f" % row["vmax"], "kernels", row.get("kernels"))
        print("   breakdown", {k: round(v, 2) for k, v in row["breakdown_ms"].items()})
        b = row["variants"]["base"]["ms_per_step"]
        for s, r in row["variants"].items():
            print("   %-16s %9.2f ms  x%.2f  act %s finite %s" % (s, r["ms_per_step"], b / r["ms_per_step"], round(r.get("active_frac", -1), 3), r["finite"]))

ce = load("results/lut_central.json")
if ce:
    print("## central force")
    print(ce["table_error"])
    for row in ce["timing"]:
        b = row["variants"]["base"]
        print(row["regime"], row["N"], "nbr/body %.1f" % row["neighbours_per_body"], {k: round(v, 2) for k, v in row["breakdown_ms"].items()},
              {k: "%.1f (x%.2f)" % (v, b / v) for k, v in row["variants"].items()})
    for k, rows in ce["accuracy"].items():
        print(k, "base", rows["base"]["200"])
        for s, r in rows.items():
            if s != "base":
                print("   ", s, {m: {a: float("%.3g" % b) for a, b in v.items() if a in ("max_dp", "mean_dp", "KE_rel", "max_dv")} for m, v in r.items() if m in ("1", "100", "200")})
