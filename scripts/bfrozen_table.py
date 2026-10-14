"""Markdown tables for the B-frozen comparison from eval_<tag>.json files.
Usage: python3 scripts/bfrozen_table.py DIR OUT.md"""
import json
import sys

MODELS = [("E_zs_bf", "E zero-shot"), ("Eg_bf", "E+g (no walls)"), ("B2_bf", "B2"), ("C_bf", "C"), ("Cfz_bf", "frozen-C"), ("Bf1", "Bf1 (SDF)"), ("Bf2", "Bf2 (SDF+pair)")]
FAMS = ("box15", "box_var", "post_circle", "post_rect", "disc", "post_poly")


def f(x, p=3):
    return "-" if x is None else f"{x:.{p}g}"


def main():
    d, out = sys.argv[1], sys.argv[2]
    R = {t: json.load(open(f"{d}/eval_{t}.json")) for t, _ in MODELS}
    L = ["| family | model | err@5 mean/med | err@10 mean/med | err@20 mean/med | err@100 mean/med | tunnel@20/@100 % | KE ratio@100 | off-grid@100 % | runaway % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        for t, name in MODELS:
            m = R[t]["families"][fam]["mean"]
            e, md = m["err"], m["err_median"]
            L.append(f"| {fam} | {name} | " + " | ".join(f"{f(e[k])}/{f(md[k])}" for k in ("5", "10", "20", "100")) +
                     f" | {f(100 * m['tunnel_model']['20'])}/{f(100 * m['tunnel_model']['100'])} | {f(m['ke_ratio']['100'])} | {f(100 * m['offgrid100'])} | {f(100 * m['runaway'])} |")
    for key in ("restitution_wall", "restitution_circle"):
        sp = list(R["Bf1"][key])
        L += ["", f"{key} (apex-after-first-reversal e; truth ~1.0)", "", "| model | " + " | ".join(f"v={v}" for v in sp) + " |", "|---|" + "---|" * len(sp)]
        for t, name in MODELS:
            L.append(f"| {name} | " + " | ".join(f(R[t][key][v].get("model_rev")) + (" (old " + f(R[t][key][v].get("model")) + ")" if R[t][key][v].get("model") is not None else "") for v in sp) + " |")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
