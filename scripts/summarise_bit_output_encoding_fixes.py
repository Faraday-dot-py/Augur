import argparse
import json

LIMIT_PASS = 1.5
LIMIT_KILL = 3.0
MARGIN = 0.02


def verdict(r):
    if r > LIMIT_KILL:
        return "KILL"
    return "PASS" if r <= LIMIT_PASS else "PARTIAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    a = ap.parse_args()
    res = json.load(open(a.results))
    runs = res["runs"]
    arms = []
    for k in runs:
        arm = k.split("/")[1]
        if arm not in arms and arm != "float":
            arms.append(arm)
    seeds = sorted({k.split("/")[0] for k in runs})
    print("seed mode arm force_err ratio_force force_verdict roll20 ratio_roll roll_verdict overall improves_vs_plain_bits")
    for seed in seeds:
        ref = res["refs"][seed]
        print(f"seed {seed} refs: zero force err 1.0, oracle cell-mean force err {ref['particle_force_rel_err']['oracle_cell_mean']:.3f}, "
              f"rollout@20 zero {ref['rollout_err']['zero_force']['err@20']:.3f} oracle {ref['rollout_err']['oracle_cell_mean']['err@20']:.3f}")
        for mode in ("centered", "global"):
            fl = runs[f"{seed}/float/{mode}"]
            base_r = runs[f"{seed}/plain-bits/{mode}"]["particle_force_rel_err"] / fl["particle_force_rel_err"]
            for arm in ["float"] + arms:
                x = runs.get(f"{seed}/{arm}/{mode}")
                if x is None:
                    continue
                rf = x["particle_force_rel_err"] / fl["particle_force_rel_err"]
                rr = x["rollout_err"]["err@20"] / fl["rollout_err"]["err@20"]
                vf, vr = verdict(rf), verdict(rr)
                overall = max(vf, vr, key=["PASS", "PARTIAL", "KILL"].index)
                imp = "n/a" if arm in ("float", "plain-bits") else ("yes" if rf < base_r - MARGIN else "no")
                print(f"{seed} {mode} {arm} {x['particle_force_rel_err']:.3f} {rf:.3f} {vf} {x['rollout_err']['err@20']:.3f} {rr:.3f} {vr} {overall} {imp}")
    print("bit accuracy (own encoding, x axis, lsb to msb), seed first")
    for seed in seeds:
        for mode in ("centered", "global"):
            for arm in arms:
                x = runs.get(f"{seed}/{arm}/{mode}")
                if x and "bit_acc_own_encoding_lsb_to_msb" in x:
                    print(seed, mode, arm, x["bit_acc_own_encoding_lsb_to_msb"]["x"])
    print("decisions", res["decisions"])


if __name__ == "__main__":
    main()
