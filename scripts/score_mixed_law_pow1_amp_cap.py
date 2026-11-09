"""Score job 7296 (pow1 amplitude cap) against the line at experiment-log.md:3787.
Usage: python scripts/score_mixed_law_pow1_amp_cap.py --eval eval.json --layers layer_stats.json --npz multi_traj.npz --review-bad N
--review-bad is the BAD count from the fresh-subagent image read; without it the verdict is withheld.
"""
import argparse
import json

import numpy as np

SEEDS = ("9000", "12000")
NEW, CMP = "pow1cap_s4738", "main_s4739"
LAWS = ("grav", "pow3", "yuk8", "yuk128", "lj")


def mean(d, case, key):
    return float(np.mean([d[case][s][key] for s in SEEDS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--review-bad", type=int, default=None)
    a = ap.parse_args()
    ev = json.load(open(a.eval))
    ly = json.load(open(a.layers))
    new, cmp_ = ev[NEW], ev[CMP]
    g = mean(new, "grav", "err20")
    print(f"gravity err@20 {g:.4f} (E {mean(ev['E'], 'grav', 'err20'):.4f}, main_s4739 {mean(cmp_, 'grav', 'err20'):.4f}); pass <= 0.0509, kill > 0.106")
    d2 = ly["dec2"]
    live = d2["mean_abs"] > 0.1
    dead = d2["dead_channels"] == d2["channels"]
    print(f"dec2 mean_abs {d2['mean_abs']:.4g} dead {d2['dead_channels']}/{d2['channels']}; live(>0.1) {live}, all dead {dead}")
    ratios = {c: mean(new, c, "force_rel") / mean(cmp_, c, "force_rel") for c in LAWS}
    for c, r in ratios.items():
        print(f"  tick-1 force {c}: new {mean(new, c, 'force_rel'):.4f} main_s4739 {mean(cmp_, c, 'force_rel'):.4f} ratio {r:.2f}")
    for c in ("charge_pow2", "pp_pow2"):
        ratios[c] = mean(new, c, "force_rel") / mean(cmp_, c, "force_rel")
        print(f"  {c}: ratio {mean(new, c, 'force_rel') / mean(cmp_, c, 'force_rel'):.2f}")
    for c in LAWS:
        print(f"  per-seed ratio {c}:", {s: round(new[c][s]["force_rel"] / cmp_[c][s]["force_rel"], 2) for s in SEEDS})
    force_ok = all(r <= 1.25 for r in ratios.values())
    z = np.load(a.npz)
    print("script bad flag (non-finite or |x|>1e3), per case, of 48:", {k.split("/")[0]: int(z[k].sum()) for k in z.files if k.startswith(tuple(f"{c}/" for c in ("grav", "pow1", "HELD_pow1.5", "HELD_yuk32", "HELD_pp_pow1.5"))) and k.endswith(f"/{NEW}/bad")})
    if a.review_bad is None:
        print("image-review BAD count not supplied: verdict withheld")
        return
    b = a.review_bad
    if not np.isfinite(g) or g > 0.106 or dead or b >= 20:
        v = "KILL"
    elif g <= 0.0509 and live and b <= 5 and force_ok:
        v = "PASS"
    else:
        v = "INCONCLUSIVE"
    print(f"image-review BAD {b}/48; force_ok {force_ok}; VERDICT {v}")


if __name__ == "__main__":
    main()
