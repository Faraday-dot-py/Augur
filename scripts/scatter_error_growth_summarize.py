import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
KS = (1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150)
BINS = ("1-10", "11-30", "31-60", "61-100", "101-150")


def f(x):
    return "-" if x is None else f"{x:.3g}"


def row(name, v, idx=None):
    idx = range(len(v)) if idx is None else idx
    print(f"{name:>26} " + " ".join(f"{f(v[i]):>8}" for i in idx))


for fam in ("expB", "cluster", "star"):
    r = d[fam]
    print(f"\n===== {fam} scenes {r['n_scenes']} N {min(r['N'])}-{max(r['N'])} secs {r['secs']:.0f}")
    print(f"{'tick':>26} " + " ".join(f"{k:>8}" for k in KS))
    print("-- exp1 free-rollout err by class (pos), and vel")
    for c, v in r["free:ep"].items():
        row("ep " + c, v["at_ks"])
    for c, v in r["free:ev"].items():
        row("ev " + c, v["at_ks"])
    print("-- TF one-step error (pos)")
    for c, v in r["tf:ep"].items():
        row("tf_ep " + c, v["at_ks"])
    for c in ("all", "nn<1", "nn>=1", "band26-32", "heavy", "light"):
        if c in r["tf:ev"]:
            row("tf_ev " + c, r["tf:ev"][c]["at_ks"])
    print("-- force error at truth states (rel) by class / radius bin")
    for c, v in r["fe_rel"].items():
        row("fe_rel " + c, v["at_ks"])
    for c, v in r["fe_rel_rbin"].items():
        row("fe_rel r" + c, v["at_ks"])
    for c, v in r["fe_abs_rbin"].items():
        row("fe_abs r" + c, v["at_ks"])
    print("-- exp2 oracle variants (pos err, all bodies)")
    for k in r:
        if k.endswith(":ep") and not k.startswith(("free", "tf")):
            row(k[:-3], r[k]["all"]["at_ks"])
    row("free", r["free:ep"]["all"]["at_ks"])
    print("-- noisy exact (state noise = scale * mean TF one-step), tf pos/vel", r["tf_mean_onestep_pos_vel"])
    for s, v in r["noisy_exact_ep"].items():
        row("noisy x" + s, v)
    print("-- exp4 IC noise amplification (|perturbed - unperturbed|)")
    for rel, v in r["ic_noise"].items():
        row(f"exact rel{rel}", v["exact"])
        row(f"model rel{rel}", v["model"])
        print(f"   init {v['init']:.3g} lyap exact {f(v['lyap_exact'])} model {f(v['lyap_model'])}")
    print("-- exp3 OOD: per tick-bin max / frac scene-ticks above train-seed envelope max (env_max, env_p99)  bins", BINS)
    for k, v in r["ood"].items():
        print(f"{k:>12} env_max {f(v['env_max']):>8} p99 {f(v['env_p99']):>8} max " + " ".join(f"{f(x):>8}" for x in v["max_by_tick_bin"]) + " | frac>env " + " ".join(f"{x:.2f}" for x in v["frac_scene_ticks_gt_env_max_by_bin"]))
    am = r["a_max_curve_free_scene_max"]
    print("a_max scene-max at ticks 1,10,20,25,30,50,100,150:", [f(am[t - 1]) for t in (1, 10, 20, 25, 30, 50, 100, 150)])
