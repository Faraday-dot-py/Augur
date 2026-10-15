import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
rows = d["scenes"]
env = d["env_max"]
T = len(rows[0])
print("env (max over scenes, ticks 0-20):", {k: round(v, 2) for k, v in env.items()})
print("scenes", len(rows), "ticks", T)
blow = [i for i, r in enumerate(rows) if max(x["a0_max"] for x in r) > 3 * env["a0_max"]]
print("blown-up scenes (a0_max > 3x env):", blow)
for k, v in d["first_exceed_env_tick_per_scene"].items():
    print(f"first tick > env, {k:>10}:", [v[i] for i in blow], " all-scene frac exceeding:", sum(x is not None for x in v) / len(v))
for i in blow:
    r = rows[i]
    t0 = next(t for t in range(len(r)) if r[t]["a0_max"] > 3 * env["a0_max"])
    print(f"-- scene {i}: runaway tick {t0}")
    for t in range(max(0, t0 - 4), min(len(r), t0 + 3)):
        x = r[t]
        print(f" t{t:2d} " + " ".join(f"{k}={x[k]:.3g}" for k in ("rho_max", "mom_max", "ak_max", "fld_max", "phi_max", "gacc_max", "a0_max", "geo_max", "tot_max", "vmax", "pmax", "cell_x", "cell_y", "cell_nearest_ball", "cell_sdf")))
print("-- non-blown scenes: ratio of max over ticks 21-59 to env max")
for k in ("rho_max", "mom_max", "fld_max", "gacc_max", "a0_max"):
    ok = [max(x[k] for x in rows[i][21:]) / env[k] for i in range(len(rows)) if i not in blow]
    print(k, "median", round(float(np.median(ok)), 2), "max", round(float(max(ok)), 2))
print("a0_max by tick, median/max over scenes:")
for t in (0, 10, 20, 21, 22, 23, 24, 25, 30, 40, 59):
    v = [r[t]["a0_max"] for r in rows if t < len(r)]
    print(t, round(float(np.median(v)), 1), round(float(max(v)), 1))
