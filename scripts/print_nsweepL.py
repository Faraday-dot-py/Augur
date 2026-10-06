"""Compact table of results/scatter_scaling_nsweepL_cell.json (extent 256, G 512, h 0.5)."""
import json

d = json.load(open("results/scatter_scaling_nsweepL_cell.json"))
for k, r in d.items():
    if "err" not in r:
        print(k, r)
        continue
    e, c, t = r["err"], r["constvel"], r.get("timing", {})
    print(f"{k} err@5/20/100 {e['5']:.3g}/{e['20']:.3g}/{e['100']:.3g} ratio20 {e['20'] / max(c['20'], 1e-12):.3f} dE/E {r['dE_rel_model']:.3g} ({r['dE_rel_truth']:.2g}) out {r['frac_out_model']:.3f} ms {t.get('tick_ms', 0):.3g} MB {t.get('peak_mem_mb', 0):.0f}")
