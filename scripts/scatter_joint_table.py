"""Side-by-side markdown tables for baseline_v2 / control / joint / int_full1 (docs/debugging/scatter-joint-scratch.md).
Usage: python scripts/scatter_joint_table.py --ctrl results/scatter_joint/NAME_ctrl --joint results/scatter_joint/NAME_joint
(prefixes of NAME_regress.json, NAME_ood_all.json, NAME_scaling_{r46,n46,n46big}.json). Baseline = Exp E files under results/, int_full1 = results/scatter_int."""
import argparse
import json
import os


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f(x):
    return "-" if x is None else f"{x:.3g}"


def row(name, vals):
    print(f"| {name} | " + " | ".join(f(v) for v in vals) + " |")


def header(title, cols):
    print(f"\n#### {title}\n\n| row | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--base", default="results/baseline_v2_regress.json")
    ap.add_argument("--int", dest="int_", default="results/scatter_int/int_full1")
    args = ap.parse_args()
    cols = ["baseline_v2", "control", "joint", "int_full1"]
    reg = [load(args.base), load(args.ctrl + "_regress.json"), load(args.joint + "_regress.json"), load(args.int_ + "_regress.json")]

    def g(r, fn):
        try:
            return fn(r) if r else None
        except KeyError:
            return None

    header("harness position error (seed mean)", cols)
    for rg in ("two_body", "expB", "bh300"):
        for k in ("5", "10", "20", "100"):
            row(f"{rg} err@{k}", [g(r, lambda r: r["regimes"][rg]["mean"]["err"][k]) for r in reg])
    header("tick-100 conservation (model; truth in last column of baseline)", cols)
    for rg in ("two_body", "expB", "bh300"):
        for k in ("dE_rel_model", "dP_model", "dL_rel_model"):
            row(f"{rg} {k}", [g(r, lambda r: r["regimes"][rg]["mean"][k]) for r in reg])
    header("1000-tick conservation rows", cols)
    for rg in ("nr100", "bh300"):
        for k in ("dE", "dL", "dP", "bound", "vir", "outgrid"):
            row(f"cons {rg} {k}", [g(r, lambda r: r["conservation"][rg]["model"][k]) for r in reg])
        row(f"cons {rg} (truth) dE", [g(reg[0], lambda r: r["conservation"][rg]["true"]["dE"])] + [None] * 3)
        row(f"cons {rg} bound_gap", [g(r, lambda r: r["conservation"][rg]["bound_gap"]) for r in reg])
        row(f"cons {rg} vir_gap", [g(r, lambda r: r["conservation"][rg]["vir_gap"]) for r in reg])
    row("bh300 vmax/c", [g(r, lambda r: r["regimes"]["bh300"]["mean"]["vmax_over_c"]) for r in reg])
    header("timing ms/tick", cols)
    for k in ("two_body", "expB", "bh300", "expB_batch1_n100"):
        row(k, [g(r, lambda r: r["timing"][k]["tick_ms"]) for r in reg])
    bat = [r.get("ood_battery") if r else None for r in reg]
    keys = [k for b in bat if b for k in b]
    header("OOD battery err@2 (mass / eps / density / other)", cols)
    for k in dict.fromkeys(keys):
        row(k, [b[k]["err@2"] if b and k in b else None for b in bat])
    row("geomean ratio to ref", [g(r, lambda r: r["ood_battery_summary"]["geomean_ratio_to_ref"]) for r in reg])
    ood = [load("results/scatter_ood.json"), load(args.ctrl + "_ood_all.json"), load(args.joint + "_ood_all.json"), load(args.int_ + "_ood_all.json")]
    if ood[2] or ood[1]:
        header("OOD all axes err@2", cols)
        ref = ood[2] or ood[1]
        for ax, pts in ref["axes"].items():
            for pn in pts:
                row(f"{ax}/{pn}", [g(o, lambda o: o["axes"][ax][pn]["mean"]["err@2.0"]) for o in ood])
    for ph, bpath, pref in (("r46", "results/scatter_scaling_worlds.json", ("R_",)), ("n46", "results/scatter_scaling_nsweep_cell.json", ("cold_", "warm_")),
                            ("n46big", "results/scatter_scaling_nsweep_cell.json", ("cold_", "warm_"))):
        sc = [load(bpath), load(f"{args.ctrl}_scaling_{ph}.json"), load(f"{args.joint}_scaling_{ph}.json"), load(f"{args.int_}_scaling_{ph}.json")]
        ref = sc[2] or sc[1] or sc[3]
        if not ref:
            continue
        for k in (5, 20):
            header(f"scaling {ph} err@{k}", cols)
            for pt in ref:
                row(pt, [g(s, lambda s: s[pt]["err"][str(k)]) for s in sc])


if __name__ == "__main__":
    main()
