"""Summarise results/far_general_eval.json: per law and method, zone ratio range, tail, sign, interior, with PASS/KILL per seed."""
import json
import sys

d = json.load(open(sys.argv[1]))
print("law | arm | per-seed [rmin..rmax | tail | sign | interior | verdict]")
for nm, row in d["laws"].items():
    b = row["baseline"]
    tag = " HELD" if row["held_out"] else ""
    print(f"{nm}{tag} | (a) base | {b['rmin']:.3f}..{b['rmax']:.3f} tail {b.get('tail', 0):.3f} sign {b['sign_ok']:.4f}")
    for mode in ("theta", "ctx"):
        cells = []
        v = []
        for m in row[mode]:
            sign_ok = m["sign_ok"] > 0.9999
            ok = m["dev"] <= 0.1 and sign_ok and m.get("tail", 0.0) <= 0.05 and m["interior"] <= 0.05
            kill = m["rmin"] < 0.5 or m["rmax"] > 2.0
            v.append("KILL" if kill else ("PASS" if ok else "partial"))
            cells.append(f"{m['rmin']:.2f}..{m['rmax']:.2f} t{m.get('tail', 0):.2f} s{m['sign_ok']:.3f} i{m['interior']:.2f} {v[-1]}")
        print(f"{nm}{tag} | {mode} | " + " ; ".join(cells))
print("\nscene (relative L1 force error, mixed-sign 1500 particles)")
for nm, r in d["scene"].items():
    print(nm, r)
