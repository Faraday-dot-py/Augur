"""Markdown table of the geometry-free gate rows and the contact-energy rows for harness JSONs.
Usage: python scripts/scatter_regress_contact_table.py name=path.json [name=path.json ...]"""
import json
import sys

from scripts import scatter_regress_contact as rc


def main():
    runs = [(a.split("=")[0], json.load(open(a.split("=", 1)[1]))) for a in sys.argv[1:]]
    print("| arm | two_body err@20 | expB err@20 | bh300 err@20 | nr100 dE@1000 | nr100 dL | bh300 dE@1000 | bh300 dL | nr100 vir_gap | bh300 vir_gap |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, r in runs:
        g, c = r["regimes"], r["conservation"]
        print(f"| {n} | {g['two_body']['mean']['err']['20']:.4g} | {g['expB']['mean']['err']['20']:.4g} | {g['bh300']['mean']['err']['20']:.4g} | "
              f"{c['nr100']['model']['dE']:.4g} | {c['nr100']['model']['dL']:.4g} | {c['bh300']['model']['dE']:.4g} | {c['bh300']['model']['dL']:.4g} | {c['nr100']['vir_gap']:.4g} | {c['bh300']['vir_gap']:.4g} |")
    print()
    names = list(rc.THRESH)
    print("| arm | " + " | ".join(f"{k} (<= {rc.THRESH[k]})" for k in names) + " | floor@20 / @100 | rest_dev wall / circle | gain_multi@50 (report only) / floor |")
    print("|---|" + "---|" * (len(names) + 3))
    for n, r in runs:
        if "contact_energy" not in r:
            continue
        s = r["contact_energy"]["summary"]
        cells = [f"{s[k]:.4g} {v}" for k, _, _, v in rc.score(s)]
        print(f"| {n} | " + " | ".join(cells) + f" | {s['e_floor@20']:.3g} / {s['e_floor@100']:.3g} | {s['rest_dev_wall']:.3g} / {s['rest_dev_circle']:.3g} | {s.get('gain_multi@50', float('nan')):.3g} / {s.get('gain_multi_floor@50', float('nan')):.3g} |")
    print()
    print("restitution e_model/e_truth, wall (v: model, truth, gain) per arm")
    for n, r in runs:
        if "contact_energy" in r:
            f = lambda d: " ".join(f"{k}:{(x['e_model'] if x['e_model'] is not None else float('nan')):.2f}/{(x['e_truth'] if x['e_truth'] is not None else float('nan')):.2f}/{x['gain']:+.2f}" for k, x in d.items())
            print(f"{n}: wall {f(r['contact_energy']['restitution_wall'])}; circle {f(r['contact_energy']['restitution_circle'])}")


if __name__ == "__main__":
    main()
