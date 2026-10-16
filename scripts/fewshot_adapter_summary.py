import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
print("bitwise", r["bitwise"], "E unchanged at end", r["e_weights_unchanged_at_end"])
f = r["free"]["eval"]


def g(ev, fam, k, n=None):
    v = ev[fam][k]["per_scene"]
    return float(np.mean(v[:n] if n else v))


hdr = "arm  fam n gated | solar err@100 err@300 err@1000 pl@300 pl@1000 belt@300 | star err@20 err@100 | solar-train err@100 pl@300 | star-train err@20 | trainloss"
print(hdr)


def row(tag, ev, meta=""):
    s = lambda k: g(ev, "solar_held", k)
    st = lambda k: g(ev, "star_held", k)
    tr = ""
    if "solar_train" in ev:
        tr = f"{g(ev, 'solar_train', 'err_100'):.3g} {g(ev, 'solar_train', 'planets_retained_300'):.2f}"
    if "star_train" in ev:
        tr += f" | {g(ev, 'star_train', 'err_20'):.3g}"
    print(f"{tag:6s} {meta:18s} | {s('err_100'):.4g} {s('err_300'):.4g} {s('err_1000'):.4g} {s('planets_retained_300'):.2f} {s('planets_retained_1000'):.2f} {s('belt_retained_300'):.2f} | {st('err_20'):.4g} {st('err_100'):.4g} | {tr}")


row("free", f, "")
for tag, a in r["arms"].items():
    row(tag, a["eval"], f"{a['family']} n{a['n']} {'g' if a['gated'] else ''} L{a['train']['loss_first50']:.3g}->{a['train']['loss_last50']:.3g}")
print("free solar-train per n:")
for n in (1, 4, 16):
    print(n, g(f, "solar_train", "err_100", n), g(f, "solar_train", "planets_retained_300", n), g(f, "star_train", "err_20", n))
print("harness (nr100_dE nr100_dL bh300_dE bh300_dL expB20 expB100 expB_dE100)")
keys = ["nr100_dE", "nr100_dL", "bh300_dE", "bh300_dL", "expB_err20", "expB_err100", "expB_dE100"]


def h(lst):
    return " ".join(f"{np.median([x[k] for x in lst]):.4g}[{min(x[k] for x in lst):.3g}-{max(x[k] for x in lst):.3g}]" for k in keys)


print("free E x3  ", h(r["free"]["harness"]))
print("OFF x3     ", h(r["off_harness"]))
for tag, a in r["arms"].items():
    print(f"{tag} ON x{len(a['harness_on'])}", h(a["harness_on"]))
