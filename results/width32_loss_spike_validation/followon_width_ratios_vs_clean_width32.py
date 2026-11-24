"""Follow-on observation (not a verdict, not an amendment of the job 7256 line): job 7256 width ratios recomputed against the mean held-out force error of the clean width-32 seeds of job 7442.
Clean = no spike epoch (spike_verdict.json), 0 skipped steps, not resumed. Refuses to compute unless the held-out evaluation is the same quantity on the same scenes as in 7256.
Usage: python results/width32_loss_spike_validation/followon_width_ratios_vs_clean_width32.py --dir results/width32_loss_spike_validation --controls results/single_law_network_sizing
"""
import argparse
import json

EVAL_SEEDS = ("4739", "4740")
SAME_ARGS_EXCEPT = {"seed", "widths", "out_dir", "ckpt_dir"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--controls", required=True)
    a = ap.parse_args()
    sv = json.load(open(f"{a.dir}/spike_verdict.json"))
    ref = {w: json.load(open(f"{a.controls}/width{w}.json")) for w in (4, 8, 16, 32, 64)}
    r32 = ref[32]
    md5_new = sorted(l.split()[0] for l in open(f"{a.dir}/job-7442.log") if "scripts/" in l)
    md5_old = sorted(l.split()[0] for l in open(f"{a.controls}/job-7256.log") if "scripts/" in l)
    assert len(md5_new) == 3 and md5_new == md5_old, (md5_new, md5_old)
    clean, checks = [], {"script_md5_equal_to_7256": True, "per_seed": {}}
    for s, v in sv["seeds"].items():
        d = json.load(open(f"{a.dir}/seed{s}/width32.json"))
        assert [str(x) for x in d["eval_seeds"]] == list(EVAL_SEEDS) == [str(x) for x in r32["eval_seeds"]], s
        diff = {k for k in set(d["args"]) | set(r32["args"]) if d["args"].get(k) != r32["args"].get(k)}
        assert diff <= SAME_ARGS_EXCEPT, (s, diff)
        floor_equal = all(d["eval"][e]["force_fp32_exact_floor_rel_l2"] == r32["eval"][e]["force_fp32_exact_floor_rel_l2"] for e in EVAL_SEEDS)
        assert floor_equal, f"seed {s}: fp32 exact-force floor differs from 7256, held-out scenes are not the same"
        checks["per_seed"][s] = {"args_differing_from_7256_width32": sorted(diff), "fp32_floor_bit_equal_to_7256_on_both_eval_seeds": floor_equal}
        if not v["spikes"] and v["skipped_steps"] == 0 and not v["resumed"]:
            clean.append(s)
    assert len(clean) >= 1, "no clean width-32 seed"
    f = {s: [json.load(open(f"{a.dir}/seed{s}/width32.json"))["eval"][e]["force_rel_l2"] for e in EVAL_SEEDS] for s in sv["seeds"]}
    refs = [sum(f[s][i] for s in clean) / len(clean) for i in range(2)]
    refm = sum(refs) / 2
    out = {"label": "FOLLOW-ON OBSERVATION, not a verdict, not an amendment of the job 7256 line", "same_quantity_checks": checks,
           "clean_train_seeds": clean, "width32_force_rel_l2_by_train_seed_7442": f,
           "width32_force_rel_l2_7256_train_seed_4738": [r32["eval"][e]["force_rel_l2"] for e in EVAL_SEEDS],
           "clean_reference_by_eval_seed": refs, "clean_reference_mean": refm, "per_width": {}}
    for w in (4, 8, 16, 32, 64):
        fw = [ref[w]["eval"][e]["force_rel_l2"] for e in EVAL_SEEDS]
        out["per_width"][str(w)] = {"force_rel_l2_by_seed": fw, "ratio_of_means_vs_clean_32": (sum(fw) / 2) / refm,
                                    "ratio_by_seed_vs_clean_32": [x / y for x, y in zip(fw, refs)]}
    json.dump(out, open(f"{a.dir}/followon_width_ratios_vs_clean_width32.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
