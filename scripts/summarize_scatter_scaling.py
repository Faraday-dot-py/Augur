"""Markdown tables + plots from results/scatter_scaling_{nsweep,worlds,stock}.json. Usage: python scripts/summarize_scatter_scaling.py [--tag SUFFIX]"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def row(r):
    t = r.get("timing", {})
    e = r["err"]
    cv = r["constvel"]
    return (f"| {r['n']} | {r['grid']} | {r['extent']:g} | {r['h']:.3g} | {e['5']:.3g} | {e['10']:.3g} | {e['20']:.3g} | {e['50']:.3g} | {e['100']:.3g} | {e['20'] / max(cv['20'], 1e-12):.2f} | "
            f"{r['selfnoise']['20']:.2g} | {r['dE_rel_model']:.3g} ({r['dE_rel_truth']:.2g}) | {r['dP_model']:.2g} | {r['first_nonfinite']} | {r['frac_out_model']:.3f} | "
            f"{t.get('tick_ms', float('nan')):.3g} | {t.get('peak_mem_mb', float('nan')):.0f} |")


HEAD = ("| N | G | extent | h | err@5 | @10 | @20 | @50 | @100 | err@20/constvel@20 | self-noise@20 | dE/E (truth) | dP | first nonfinite | frac out | ms/tick | peak MB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def slope(ns, ts):
    p = np.polyfit(np.log(ns), np.log(ts), 1)
    return float(p[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ns = json.load(open(f"results/scatter_scaling_nsweep{args.tag}.json"))
    out = []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for fam in ("cold", "warm"):
        pts = sorted([v for k, v in ns.items() if k.startswith(fam) and "err" in v], key=lambda r: r["n"])
        out.append(f"\n### N sweep, extent 64, G 128, {fam}\n\n{HEAD}")
        out += [row(r) for r in pts if "selfnoise" in r]
        N = np.array([r["n"] for r in pts])
        ax[0].loglog(N, [r["err"]["20"] for r in pts], "o-", label=f"{fam} err@20")
        ax[0].loglog(N, [r["constvel"]["20"] for r in pts], "x--", label=f"{fam} const-vel@20")
        ax[1].semilogx(N, [r["err"]["20"] / max(r["constvel"]["20"], 1e-12) for r in pts], "o-", label=fam)
        if fam == "cold":
            tt = [(r["n"], r["timing"]["tick_ms"]) for r in pts if "tick_ms" in r.get("timing", {})]
            n_, t_ = zip(*tt)
            ax[2].loglog(n_, t_, "o-", label="ms/tick (chunked pair term)")
            big = [(a, b) for a, b in tt if a >= 1e4]
            out.append(f"\nlog-log slope of ms/tick vs N: all N {slope(*zip(*tt)):.2f}; N>=1e4 {slope(*zip(*big)):.2f}")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("err@20 (sim units)")
    ax[0].legend(fontsize=7)
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("err@20 / const-vel err@20")
    ax[1].legend()
    ax[2].set_xlabel("N")
    ax[2].set_ylabel("ms/tick")
    try:
        st = json.load(open(f"results/scatter_scaling_stock.json"))
        for v in ("stock", "chunked"):
            pts = sorted([x for k, x in st.items() if k.startswith(v) and "tick_ms" in x], key=lambda x: x["n"])
            ax[2].loglog([x["n"] for x in pts], [x["tick_ms"] for x in pts], "s--", label=f"{v} (stock-phase)")
        out.append("\n### Stock vs chunked pair term (B=1, clipped-gaussian cold scene, ticks 20/5)\n\n| N | stock ms/tick | stock peak MB | chunked ms/tick | chunked peak MB |\n|---|---|---|---|---|")
        for n in sorted({x["n"] for x in st.values()}):
            a, b = st.get(f"stock_{n}", {}), st.get(f"chunked_{n}", {})
            out.append(f"| {n} | {a.get('tick_ms', a.get('error', ''))} | {a.get('peak_mem_mb', '')} | {b.get('tick_ms', b.get('error', ''))} | {b.get('peak_mem_mb', '')} |")
    except FileNotFoundError:
        pass
    ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"videos/scatter_scaling_nsweep{args.tag}.png", dpi=110)
    wd = json.load(open(f"results/scatter_scaling_worlds{args.tag}.json"))
    for fam, title in (("R", "R: resolution only (N 300, scene fixed, extent 64)"), ("D", "D: domain only (h 0.5, scene fixed, N 300)"),
                       ("S", "S: scene scaled with box, G 128 (h varies)"), ("S2", "S2: scene scaled with box, h 0.5, N 300"), ("J", "J: joint constant density (h 0.5, N 300 (L/64)^2)")):
        pts = [v for v in wd.values() if v.get("family") == fam and "err" in v]
        out.append(f"\n### {title}\n\n{HEAD}")
        out += [row(r) for r in pts if "selfnoise" in r]
    open(f"results/scatter_scaling_tables{args.tag}.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
