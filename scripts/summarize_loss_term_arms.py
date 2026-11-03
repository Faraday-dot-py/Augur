"""Scores the tier-0 loss-term fine-tune arms against the pre-registered lines in docs/debugging/experiment-log.md (2026-10-26 local-work-bias and clump-centre).
Usage: python scripts/summarize_loss_term_arms.py <dir with <tag>_{eval,diag,hier,tiling,hext}.json> <control tag> <arm tag> [<arm tag> ...] [--out file.json]"""
import json
import os
import sys

BASE = "results/loss_term_arms"
SEEDS = ("9100", "9200", "9300")


def dense_work(diag, arm, dens="3", ticks=None):
    num = den = 0.0
    for sd in SEEDS:
        for tg, r in diag[sd][arm]["density"][dens].items():
            if ticks and tg not in ticks:
                continue
            num += r["work_per_particle"] * r["n"]
            den += r["n"]
    return num / den


def overall_work(diag, arm):
    num = den = 0.0
    for sd in SEEDS:
        for r in diag[sd][arm]["all"]["0"].values():
            num += r["work_per_particle"] * r["n"]
            den += r["n"]
    return num / den


def beta_mid(hext):
    out = []
    for b in (2, 3):
        g = [r for r in hext["rows"] if r["scene"] == "mid" and r["mode"] == "ext" and r["bin"] == b]
        out.append(sum(r["numc"] for r in g) / max(sum(r["denc"] for r in g), 1e-12))
    return out


def plain20(ev):
    v = [r["err5_10_20"][2] for k, r in ev["res"].items() if k.startswith("plain_")]
    return sum(v) / len(v)


def load(d, tag, kind):
    p = os.path.join(d, f"{tag}_{kind}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def metrics(d, tag):
    m = {}
    ev, dg, hr, tl, hx = (load(d, tag, k) for k in ("eval", "diag", "hier", "tiling", "hext"))
    if ev:
        m["plain_err20"] = plain20(ev)
    if dg:
        m["dense_work"] = dense_work(dg, "OR")
        m["dense_work_ticks10_30"] = dense_work(dg, "OR", ticks=("10-30",))
        m["overall_work"] = overall_work(dg, "OR")
    if hr:
        o = hr["OR"]
        m["force_rel"] = sum(o[s]["force_rel"] for s in SEEDS) / 3
        m["com_err"] = {s: o[s]["com_err"] for s in SEEDS}
        m["hier_err20"] = sum(o[s]["err@20"] for s in SEEDS) / 3
        m["hier_dE"] = sum(o[s]["dE_model"] for s in SEEDS) / 3
    if tl:
        k = [x for x in tl if x.endswith("@128")][0]
        m["tiled_dE100"] = tl[k]["dE@100_model"]
        m["tiled_err20"] = tl[k]["err@20"]
    if hx:
        m["beta_mid_8_16_16_32"] = beta_mid(hx)
    return m


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out:
        a.remove(out)
    d, ctrl, arms = a[0], a[1], a[2:]
    bd = json.load(open(f"{BASE}/baseline_hier_diagnosis.json"))
    bh = json.load(open(f"{BASE}/baseline_hier_eval_matched60k.json"))
    base = {"BL_dense_work": dense_work(bd, "BL"), "BL_overall_work": overall_work(bd, "BL"),
            "matched60k_dense_work": dense_work(bd, "OR"), "matched60k_overall_work": overall_work(bd, "OR"),
            "matched60k_com_err": {s: bh["OR"][s]["com_err"] for s in SEEDS}, "matched60k_force_rel": sum(bh["OR"][s]["force_rel"] for s in SEEDS) / 3,
            "BL_force_rel": sum(bh["BL"][s]["force_rel"] for s in SEEDS) / 3,
            "matched60k_beta_mid": beta_mid(json.load(open(f"{BASE}/baseline_hext_matched60k.json"))),
            "matched60k_tiled_dE100": json.load(open(f"{BASE}/baseline_tiling_matched60k.json"))["S3@128"]["dE@100_model"]}
    res = {"baseline": base, ctrl: metrics(d, ctrl)}
    c = res[ctrl]
    for t in arms:
        m = res[t] = metrics(d, t)
        v = {}
        if "dense_work" in m:
            v["dense_work_ratio_vs_BL"] = abs(m["dense_work"]) / abs(base["BL_dense_work"])
            v["dense_work_ratio_control_vs_BL"] = abs(c["dense_work"]) / abs(base["BL_dense_work"])
        if "plain_err20" in m and "plain_err20" in c:
            v["plain_err20_over_control"] = m["plain_err20"] / c["plain_err20"]
        if "force_rel" in m and "force_rel" in c:
            v["force_rel_over_control"] = m["force_rel"] / c["force_rel"]
        if "com_err" in m:
            v["com_err_over_matched60k"] = {s: m["com_err"][s] / base["matched60k_com_err"][s] for s in SEEDS}
            v["com_err_over_control"] = {s: m["com_err"][s] / c["com_err"][s] for s in SEEDS}
        m["ratios"] = v
    for k, v in res.items():
        print(k, json.dumps(v, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
