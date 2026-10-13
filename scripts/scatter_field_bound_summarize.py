import json
import sys

r = json.load(open(sys.argv[1]))
print("variant | C contact runaway | C contact err@20/@100 | E contact err@20/@100 | C grav expB err@20/@100/runaway | E grav expB err@20/@100 | E grav cluster err@100 | C nr100 dE | E nr100 dE | C bh300 dE | E bh300 dE")
for k, v in r.items():
    cr = v["C_contact_runaway"]
    ce, ee = v["C_contact_c_id"]["err"], v["E_contact_c_id"]["err"]
    g = v["C_grav_expB"]
    f = lambda x: f"{x:.3g}"
    cons = lambda n, reg: f(v[n + "_cons"][reg]["model"]["dE"]) if n + "_cons" in v else "-"
    print(k, "|", f(cr["frac_gt_200"]), "|", f(ce["20"]), f(ce["100"]), "|", f(ee["20"]), f(ee["100"]), "|", f(g["err"]["20"]), f(g["err"]["100"]), f(g["runaway"]), "|",
          f(v["E_grav_expB"]["err"]["20"]), f(v["E_grav_expB"]["err"]["100"]), "|", f(v["E_grav_cluster"]["err"]["100"]), "|", cons("C", "nr100"), "|", cons("E", "nr100"), "|", cons("C", "bh300"), "|", cons("E", "bh300"))
print()
for k, v in r.items():
    print(k, "dE@20/@100 C", {a: round(b, 2) for a, b in v["C_contact_c_id"]["dE_model"].items()}, "E", {a: round(b, 2) for a, b in v["E_contact_c_id"]["dE_model"].items()},
          "wall rest C", {s: (None if x["model"] is None else round(x["model"], 2)) for s, x in v["C_rest_wall"].items()},
          "C tunnel@100", round(v["C_contact_c_id"]["tunnel_model"]["100"], 3), "E gravity runaway", v["E_grav_expB"]["runaway"], v["E_grav_cluster"]["runaway"])
