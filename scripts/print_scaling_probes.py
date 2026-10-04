"""Print force-error probes (rel. to exact, acceleration-weighted) from scatter_scaling JSONs."""
import json
import sys

d = json.load(open(sys.argv[1]))
keys = sys.argv[2:]
for k in keys:
    r = d[k]
    ps = r["per_scene"][0]["probes"]
    print(k, "N", r["n"], "G", r["grid"], "L", r["extent"])
    for t, p in ps.items():
        print(f"   tick {t}: wmean_rel {p['wmean_rel']:.3f} nopp {p['wmean_rel_nopp']:.3f} core/mid/halo {p['wmean_rel_core']:.3f}/{p['wmean_rel_mid']:.3f}/{p['wmean_rel_halo']:.3f} cos {p['cos']:.3f} |a| core/halo {p['amag_core']:.3g}/{p['amag_halo']:.3g}")
