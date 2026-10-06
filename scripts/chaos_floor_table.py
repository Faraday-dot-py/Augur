"""Print the chaos-floor tables from results/chaos_{bh,expb}.json: floors, model error, ratios, equivalent perturbation, growth, regime splits."""
import json
import sys

import numpy as np

KS = (5, 10, 20, 50, 100)
RELS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def val(res, g, key):
    return [v[0] for v in res[g][key]]


def main(path):
    r = json.load(open(path))
    s = r["summary"]
    model = val(s, "all", "model")
    print(f"## {path} ({r['n_scenes']} scenes)")
    print("| k | model | f32 | sub8 | sub2 | " + " | ".join(f"pert{x:g}" for x in RELS) + " | model/f32 | model/sub8 | model/pert1e-5 | eq. rel pert (1e-5 x model/pert1e-5) |")
    print("|" + "---|" * (10 + len(RELS)))
    for i, k in enumerate(KS):
        f = lambda key: val(s, "all", key)[i]
        pf = [f(f"pert_{x:g}") for x in RELS]
        print(f"| {k} | {model[i]:.3g} | {f('f32'):.2g} | {f('sub8'):.2g} | {f('sub2'):.2g} | " + " | ".join(f"{x:.2g}" for x in pf)
              + f" | {model[i] / f('f32'):.3g} | {model[i] / f('sub8'):.3g} | {model[i] / pf[1]:.3g} | {1e-5 * model[i] / pf[1]:.2g} |")
    print("\nlinearity of floor (pert1e-4 / (100 x pert1e-6)) per k:", [round(val(s, 'all', 'pert_0.0001')[i] / (100 * val(s, 'all', 'pert_1e-06')[i]), 2) for i in range(5)])
    c = r["mean_curves"]["pert_1e-05"]
    p0 = 1e-5
    print("growth of pert 1e-5 curve (err@k / err@1):", [round(c[k - 1] / c[0], 2) for k in KS], "err@1", c[0])
    t = np.arange(1, len(c) + 1) * r["dt"]
    for lo, hi in ((0, 20), (20, 50), (50, 100)):
        sl = np.polyfit(t[lo:hi], np.log(c[lo:hi]), 1)[0]
        print(f"  local growth rate of ln err, ticks {lo + 1}-{hi}: {sl:.3f} per time unit")
    print("\nregime splits (err@k: model | f32 | sub8 | pert1e-5 ; n scenes with members):")
    for g in s:
        if g == "all":
            continue
        for key in ("model", "f32", "sub8", "pert_1e-05"):
            print(f"  {g:>13} {key:>10}", [None if v[0] is None else float(f"{v[0]:.3g}") for v in s[g][key]], [v[1] for v in s[g][key]][0])
    print("frac bodies close by k:", r["frac_bodies_close_by_k"])
    print("model self-perturbation 1e-5 err@k:", [float(f"{v:.3g}") for v in val(s, "all", "model_selfpert_1e-05")])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
