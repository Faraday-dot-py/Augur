"""Summaries of an escape_roll.py output directory (N <= 2e4 with truth.npz for per-body errors; any N for series). Prints tables; writes <dir>/summary.json.
Usage: python scripts/escape_analyze.py results/escape/n2e4_s4738 [arm ...]"""
import json
import os
import sys

import numpy as np

WIN = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 700), (700, 1000)]
TK = (20, 50, 100, 200, 300, 500, 1000)


def load(d, name):
    r = json.load(open(f"{d}/{name}.json"))
    z = np.load(f"{d}/{name}.npz")
    return r, z


def at(rows, t, key):
    for r in rows:
        if r["t"] == t:
            return r.get(key)
    return None


def main():
    d = sys.argv[1]
    names = sys.argv[2:] or sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and f != "summary.json")
    if "truth" in names:
        names.remove("truth")
        names = ["truth"] + names
    data = {n: load(d, n) for n in names}
    truth = data.get("truth")
    out = {}
    print(f"== {d}")
    print("series: frac outside 64-box | frac outside own domain | r99 | n r>70 (of tracked) at t=100/300/500/1000")
    for n in names:
        rows = data[n][0]["rows"]
        s = lambda k, f: " / ".join(f.format(at(rows, t, k)) if at(rows, t, k) is not None else "-" for t in (100, 300, 500, 1000))
        print(f"{n:6s} out32 {s('frac_out32', '{:.3f}')} | dom {s('frac_outdom', '{:.3f}')} | r99 {s('r99', '{:.0f}')} | r70 {s('n_r70', '{}')}/{rows[-1]['n_track']}")
        rev = []
        for lo, hi in WIN:
            o = sum(r["out"] for r in rows if lo < r["t"] <= hi)
            v = sum(r["rev"] for r in rows if lo < r["t"] <= hi)
            rev.append(f"{lo}-{hi}: {100 * v / max(o, 1):.2f}% ({v}/{o})")
        print("       reversal band 28-32: " + " | ".join(rev))
        out[n] = {"final": rows[-1]}
    print("core: own-centre position / r50 / n_core10 / set COM, at t=100/200/300/400/500/700/1000 ; distance of set COM from truth set COM")
    tr = truth[0]["rows"] if truth else None
    for n in names:
        rows = data[n][0]["rows"]
        line = []
        for t in (100, 200, 300, 400, 500, 700, 1000):
            c, r50, nc, sc = at(rows, t, "centre"), at(rows, t, "r50"), at(rows, t, "n_core10"), at(rows, t, "set_com")
            if c is None:
                continue
            dd = ""
            if tr is not None and sc is not None and at(tr, t, "set_com") is not None:
                dd = f" dTr {np.linalg.norm(np.array(sc) - np.array(at(tr, t, 'set_com'))):.1f}"
            line.append(f"t{t}: c=({c[0]:.1f},{c[1]:.1f}) r50 {r50:.2f} n10 {nc}" + (f" set=({sc[0]:.1f},{sc[1]:.1f}){dd}" if sc else ""))
        print(f"{n:6s} " + " ; ".join(line))
    print("momentum: |P_all| / |P_inbox| / |core_P| at t=100/300/500/1000")
    for n in names:
        rows = data[n][0]["rows"]
        f = lambda k, t: (f"{np.linalg.norm(at(rows, t, k)):.3g}" if at(rows, t, k) is not None else "-")
        print(f"{n:6s} " + " ".join(f"t{t}: {f('P_all', t)}/{f('P_inbox', t)}/{f('core_P', t)}" for t in (100, 300, 500, 1000)))
    if truth is not None:
        zt = truth[1]
        tt = list(zt["snap_t"])
        print("per-body position error vs truth (mean |dx|), by TRUTH class at that tick: in-box (<32) | outside box | r>70 escapees ; and ratio escapee/in-box")
        for n in names:
            if n == "truth":
                continue
            z = data[n][1]
            tn = list(z["snap_t"])
            parts = []
            res = {}
            for t in TK:
                if t not in tt or t not in tn:
                    continue
                pt, pm = zt["snap_pos"][tt.index(t)].astype(np.float64), z["snap_pos"][tn.index(t)].astype(np.float64)
                e = np.linalg.norm(pt - pm, axis=1)
                c = np.abs(pt).max(1)
                rr = np.linalg.norm(pt, axis=1)
                ein, eout, eesc = (e[c < 32].mean() if (c < 32).any() else np.nan), (e[c >= 32].mean() if (c >= 32).any() else np.nan), (e[rr > 70].mean() if (rr > 70).any() else np.nan)
                res[t] = {"in": float(ein), "out": float(eout), "esc": float(eesc), "n_in": int((c < 32).sum()), "n_esc": int((rr > 70).sum())}
                parts.append(f"t{t}: {ein:.2f} | {eout:.2f} | {eesc:.2f} (x{eesc / ein:.2f}; n_esc {int((rr > 70).sum())})")
            out[n]["err"] = res
            print(f"{n:6s} " + " ; ".join(parts))
        print("radial distribution vs truth: fraction r>70 / r>100 / r>200 at t=100/300/500 (model | truth)")
        for n in names:
            if n == "truth":
                continue
            z = data[n][1]
            tn = list(z["snap_t"])
            line = []
            for t in (100, 300, 500):
                if t not in tt or t not in tn:
                    continue
                pt, pm = zt["snap_pos"][tt.index(t)], z["snap_pos"][tn.index(t)]
                rt, rm = np.linalg.norm(pt, axis=1), np.linalg.norm(pm, axis=1)
                line.append(f"t{t}: " + "/".join(f"{(rm > q).mean():.3f}|{(rt > q).mean():.3f}" for q in (70, 100, 200)))
            print(f"{n:6s} " + " ; ".join(line))
    for n in names:
        pr = data[n][0]["probes"]
        if not pr:
            continue
        print(f"probes {n}: t, core15 (n, bias norm, mean exact a, mean model a, rms rel err, tan/rad rms err) | inbox net force model vs exact | abar_raw | parts mean(core15)")
        for p in pr:
            c = p["cls"].get("core15")
            if not c:
                continue
            pa = p.get("parts", {}).get("core15")
            ps = ""
            if pa:
                ps = f" parts k={np.round(pa['kernel_mean'], 2).tolist()} u={np.round(pa['unet_mean'], 2).tolist()} p={np.round(pa['pair_mean'], 2).tolist()} ex={np.round(pa['exact_mean'], 2).tolist()}"
            print(f"  t{p['t']} n{c['n']} bias {c['bias_norm']:.3g} ex {np.round(c['exact_mean'], 3).tolist()} mod {np.round(c['model_mean'], 3).tolist()} rel {c['rms_rel']:.3f} "
                  f"tan {c['tan_rms_err']:.3g} rad {c['rad_rms_err']:.3g} | inbox net mod {np.round(p['net_raw_model_inbox'], 1).tolist()} ex {np.round(p['net_exact_inbox'], 1).tolist()} "
                  f"nin {p['n_inbox']} | abar {np.round(p['abar_raw'], 4).tolist()}{ps}")
    json.dump(out, open(f"{d}/summary.json", "w"))


if __name__ == "__main__":
    main()
