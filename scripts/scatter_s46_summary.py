"""Markdown rows for one review-fix run vs the frozen Exp E baseline rows.
Usage: python scripts/scatter_s46_summary.py NAME [--dir results/scatter_s46]
Reads <dir>/NAME_ood_*.json (baseline results/scatter_ood.json), NAME_scaling_{r46,n46,n46big}.json (baseline results/scatter_scaling_worlds.json for R_G*,
results/scatter_scaling_nsweep_cell.json for cold_/warm_N), NAME_regress.json (baseline results/scatter_baseline.json) incl. the OOD battery."""
import argparse
import glob
import json
import os


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x):
    return "-" if x is None else f"{x:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--dir", default="results/scatter_s46")
    args = ap.parse_args()
    d, n = args.dir, args.name
    reg = load(f"{d}/{n}_regress.json")
    base = load("results/scatter_baseline.json")
    if reg:
        print("\n#### harness (seed-mean err@5/10/20, baseline -> run)\n\n| regime | err@5 | err@10 | err@20 | err@50 | err@100 |\n|---|---|---|---|---|---|")
        for r in ("two_body", "expB", "bh300"):
            cells = [f"{fmt(base['regimes'][r]['mean']['err'][k])} -> {fmt(reg['regimes'][r]['mean']['err'][k])}" for k in ("5", "10", "20", "50", "100")]
            print(f"| {r} | " + " | ".join(cells) + " |")
        print("\n| timing | ms/tick base -> run | peak MB base -> run |\n|---|---|---|")
        for k in base["timing"]:
            print(f"| {k} | {fmt(base['timing'][k]['tick_ms'])} -> {fmt(reg['timing'][k]['tick_ms'])} | {fmt(base['timing'][k]['peak_mem_mb'])} -> {fmt(reg['timing'][k]['peak_mem_mb'])} |")
        if "ood_battery" in reg:
            print(f"\n#### OOD battery err@2 (run; geomean ratio to ref {fmt(reg['ood_battery_summary']['geomean_ratio_to_ref'])})\n\n| point | err@2 | floor@2 |\n|---|---|---|")
            for k, v in reg["ood_battery"].items():
                print(f"| {k} | {fmt(v['err@2'])} | {fmt(v['floor@2'])} |")
    bo = load("results/scatter_ood.json")
    for path in sorted(glob.glob(f"{d}/{n}_ood_*.json")):
        res = load(path)
        print(f"\n#### OOD {os.path.basename(path)} (err@2 time units; baseline -> run; floor@2)\n\n| axis | point | err@.5 | err@1 | err@2 | err@5 | floor@2 | nonfinite |\n|---|---|---|---|---|---|---|---|")
        for ax, pts in res["axes"].items():
            for pn, v in pts.items():
                if "mean" not in v:
                    print(f"| {ax} | {pn} | error | | | | | |")
                    continue
                m = v["mean"]
                b = bo["axes"].get(ax, {}).get(pn, {}).get("mean") if bo else None
                cells = [f"{fmt(b[f'err@{t}']) if b else '-'} -> {fmt(m[f'err@{t}'])}" for t in ("0.5", "1.0", "2.0", "5.0")]
                print(f"| {ax} | {pn} | " + " | ".join(cells) + f" | {fmt(m['floor@2.0'])} | {m['first_nonfinite_tick']} |")
    bw, bn = load("results/scatter_scaling_worlds.json"), load("results/scatter_scaling_nsweep_cell.json")
    for ph in ("r46", "n46", "n46big"):
        res = load(f"{d}/{n}_scaling_{ph}.json")
        if not res:
            continue
        print(f"\n#### scaling {ph} (baseline -> run)\n\n| point | err@5 | err@20 | err@5/constvel@5 | frac out |\n|---|---|---|---|---|")
        for k, v in res.items():
            if "err" not in v:
                print(f"| {k} | {v.get('error')} | | | |")
                continue
            b = (bw if k.startswith("R_") else bn).get(k) if (bw if k.startswith("R_") else bn) else None
            b = b if b and "err" in b else None
            print(f"| {k} | {fmt(b['err']['5']) if b else '-'} -> {fmt(v['err']['5'])} | {fmt(b['err']['20']) if b else '-'} -> {fmt(v['err']['20'])} | "
                  f"{fmt(b['err']['5'] / b['constvel']['5']) if b else '-'} -> {fmt(v['err']['5'] / max(v['constvel']['5'], 1e-12))} | {fmt(v['frac_out_model'])} |")


if __name__ == "__main__":
    main()
