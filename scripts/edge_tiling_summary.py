"""Summarise outer-edge ('outer<8') and seam force error from the unmerged tiling force results (scatter_tiling_force.json, branch worktree-agent-a6dc0f9444b821098):
tiled vs exact and the single grid vs exact, per family, held seeds, ticks 0/5/10/20, variants g16_fx_far and t32_g16_fx_far."""
import json
import sys

from statistics import mean

d = json.load(open(sys.argv[1]))
for key, rows in d.items():
    if not key.endswith("/held"):
        continue
    for var in ("g16_fx_far", "t32_g16_fx_far", "g16_fx"):
        o, s, i = [], [], []
        so = []
        for r in rows:
            v = r[var]["vs_exact"]
            if v["outer<8"]["rel"] is not None:
                o.append(v["outer<8"]["rel"])
            if v["seam<8"]["rel"] is not None:
                s.append(v["seam<8"]["rel"])
            if v["interior"]["rel"] is not None:
                i.append(v["interior"]["rel"])
            sv = r.get("single_vs_exact_by_g16_class", {}).get("outer<8", {}).get("rel")
            if sv is not None and var == "g16_fx":
                so.append(sv)
        m = lambda x: f"{mean(x):.3f} (n={len(x)})" if x else "-"
        print(f"{key} {var}: tiled-vs-exact outer<8 {m(o)} seam<8 {m(s)} interior {m(i)}" + (f" | single-vs-exact outer<8 {m(so)}" if so else ""))
