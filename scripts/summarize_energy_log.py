import re, sys
sec = None
for line in open(sys.argv[1]):
    line = re.sub(r"np\.float32\(([^)]*)\)", r"\1", line.rstrip())
    if line.startswith("==="):
        sec = line; print(line); continue
    if line.startswith("wall {"):
        d = eval(line[5:])
        if d["wall"] == "y_wall":
            print(f"  ywall s={d['speed_in']:3d} restit {d['model_ratio']:.2f}")
        else:
            print(f"  xfloor s={d['speed_in']:3d} dE {d['model_dE']:8.1f}")
    elif line.startswith("pair {"):
        d = eval(line[5:])
        if d["offset"] == 0.0 and d["speed_in"] in (3, 6, 10, 20):
            print(f"  pair head-on s={d['speed_in']:3d} relKE {d['model_relKE_ratio']:.2f} pmom {d['model_pmom_err']:.1f}")
    elif line.startswith("hidden_reset"):
        import json
        d = json.loads(line[len("hidden_reset "):])
        print("  n20 4ball E carried", {k: round(v, 1) for k, v in d["carried"].items()}, "truth", round(d["truth"]["100"], 1), "err", {k: round(v, 2) for k, v in d["carried_err"].items() if k in ("5", "10", "20")})
    elif re.match(r"^\s+(20|30|50|100|200) tE=", line) or line.startswith("== nb="):
        print(" ", line[:150])
    elif line.startswith("it 1500") or line.startswith("it 3000") or line.startswith("it 100 "):
        print(" ", line)
