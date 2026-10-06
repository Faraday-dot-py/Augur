"""Print tick-1 force error table from results/scatter_scaling_fprobe.json."""
import json

d = json.load(open("results/scatter_scaling_fprobe.json"))
for k, v in d.items():
    p = v["probe"]
    print(k, f"cells {v.get('sigma_cells', '')} wmean_rel {p['wmean_rel']:.3f} nopp {p['wmean_rel_nopp']:.3f} core/mid/halo {p['wmean_rel_core']:.3f}/{p['wmean_rel_mid']:.3f}/{p['wmean_rel_halo']:.3f} cos {p['cos']:.3f} amag {p['amag_core']:.3g}/{p['amag_halo']:.3g}")
