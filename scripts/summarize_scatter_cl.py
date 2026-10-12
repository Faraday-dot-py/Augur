"""Table of past-horizon behaviour and harness rows for scatter_cl results. Usage: python scripts/summarize_scatter_cl.py results/scatter_cl base cl_truth cl_relabel ..."""
import json
import sys

TICKS = (5, 20, 50, 100)


def main():
    d, names = sys.argv[1], sys.argv[2:]
    print("| model | " + " | ".join(f"err@{t}" for t in TICKS) + " | field rms@100 | a_rms/true@100 | a_max/true@100 | pos_x/pos_y rms@100 | first nonfinite | relabel err@20 off 0/5/10/20 |")
    print("|" + "---|" * 9)
    for n in names:
        r = json.load(open(f"{d}/{n}_past.json"))
        ss = list(r["seeds"].values())
        mean = lambda k, t: sum(s["series"][k][t - 1] for s in ss) / len(ss)
        errs = [f"{mean('err', t):.4f}" if t < 50 else f"{mean('err', t):.3f}" for t in TICKS]
        arat = mean("a_rms", 100) / mean("true_a_rms", 100)
        amax = mean("a_max", 100) / mean("true_a_max", 100)
        fn = [s["first_nonfinite_tick"] for s in ss]
        rl = " / ".join(f"{r['relabel_err20'][k]:.4f}" for k in ("0", "5", "10", "20"))
        print(f"| {n} | " + " | ".join(errs) + f" | {mean('field_rms', 100):.3f} | {arat:.3f} | {amax:.3f} | {mean('pos_x', 100):.3f}/{mean('pos_y', 100):.3f} | {fn} | {rl} |")
    print()
    print("| model | regime | err@5 | @10 | @20 | @50 | @100 | dE@100 | dL@100 |")
    print("|" + "---|" * 9)
    for n in names:
        try:
            r = json.load(open(f"{d}/{n}_regress.json"))
        except FileNotFoundError:
            continue
        for reg, v in r["regimes"].items():
            m = v["mean"]
            print(f"| {n} | {reg} | " + " | ".join(f"{m['err'][k]:.4g}" for k in ("5", "10", "20", "50", "100")) + f" | {m['dE_rel_model']:.3g} | {m['dL_rel_model']:.3g} |")
        for reg, v in r.get("conservation", {}).items():
            print(f"| {n} | cons {reg}@1000 | dE {v['model']['dE']:.4g} (true {v['true']['dE']:.2g}) | dL {v['model']['dL']:.3g} | bound {v['model']['bound']:.3f}/{v['true']['bound']:.3f} | vir_gap {v['vir_gap']:.3f} | | | |")


if __name__ == "__main__":
    main()
