"""Summarise results/orbit_<tag>.json: separation / retention every --every steps and diagnostics rows."""
import json
import sys

d = json.load(open(sys.argv[1]))
every = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
print("args", {k: d["args"][k] for k in ("mode", "n", "steps", "dt", "force", "sigma", "d", "vfac", "c", "ratio") if k in d["args"]}, "wall_s", round(d["wall_s"]))
for s, t in zip(d["track_steps"], d["tracks"]):
    if s % every == 0:
        print(f"step {s:6d} sep {t['sep']:8.1f} a_r4 {t['a_r4']:.3f} a_r8 {t['a_r8']:.3f} a_own {t['a_own']:.3f} b_r4 {t['b_r4']:.3f} b_r8 {t['b_r8']:.3f} b_own {t['b_own']:.3f}")
for g in d["diags"]:
    print(f"diag {g['step']:6d} E {g['E']:.5g} P {g['P']:.2e} L {g['L']:.5g} Epos {g['frac_E_pos']:.4f} rq {[round(x, 1) for x in g['rq']]} clamp {g['clamped_frac']:.3f}" + (f" beyondRs {g['frac_beyond_Rs']:.5f} vmax {g['max_speed']:.1f}" if "R_s" in g else ""))
