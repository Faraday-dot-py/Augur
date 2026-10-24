"""Summaries for scripts/hetero_tier_checks.py outputs. Usage: python scripts/summarize_hetero.py <dir> <tag>  (reads <dir>/hetero_{a,b,c}_<tag>.json)"""
import json
import sys
from collections import defaultdict

import numpy as np

BINS = ("0-4", "4-8", "8-16", "16-32")
HELD = (9000, 12000)


def load(d, ph, tag):
    return json.load(open(f"{d}/hetero_{ph}_{tag}.json"))["rows"]


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summ_a(rows, held, scene):
    rows = [r for r in rows if r.get("scene") == scene]
    rs = [r for r in rows if "kind" not in r and ((r["seed"] in HELD) == held)]
    print(f"== (a) [{scene}] force bias, {'held-out' if held else 'dev'}: beta=<d.F>/<F.F> (d=model-isolated truth, F=full-isolated), +-SE over windows")
    print("tick bin n  beta  se | rel_err_vs_iso rel_err_vs_full |F|/|a| | mean d_x d_y (per body)")
    for t in (0, 5, 10, 20):
        for b in range(4):
            g = [r for r in rs if r["tick"] == t and r["bin"] == b]
            if not g:
                continue
            n = sum(r["n"] for r in g)
            beta = sum(r["num"] for r in g) / max(sum(r["den"] for r in g), 1e-12)
            bl = [r["num"] / r["den"] for r in g if r["den"] > 1e-9]
            print(f"{t:3d} {BINS[b]:>5} {n:6d} {beta:+.3f} {se(bl):.3f} | {sum(r['e_iso'] for r in g)/sum(r['a_iso'] for r in g):.4f} "
                  f"{sum(r['e_full'] for r in g)/sum(r['a_full'] for r in g):.4f} {sum(r['F'] for r in g)/sum(r['a_iso'] for r in g):.4f} | "
                  f"{sum(r['dx'] for r in g)/n:+.2e} {sum(r['dy'] for r in g)/n:+.2e}")
    print("-- ticks 5/10/20 pooled (excludes cold-field tick 0): bin n beta se(window-level) |F|/|a| err_vs_iso")
    for b in range(4):
        g = [r for r in rs if r["tick"] in (5, 10, 20) and r["bin"] == b]
        if g:
            bl = [r["num"] / r["den"] for r in g if r["den"] > 1e-9]
            print(f"{BINS[b]:>5} {sum(r['n'] for r in g):6d} {sum(r['num'] for r in g)/max(sum(r['den'] for r in g),1e-12):+.3f} {se(bl):.3f} "
                  f"{sum(r['F'] for r in g)/sum(r['a_iso'] for r in g):.3f} {sum(r['e_iso'] for r in g)/sum(r['a_iso'] for r in g):.3f}")
    if held:
        print("-- per-seed pooled beta (ticks pooled), bins 8-16 and 16-32")
        for sd in HELD:
            for b in (2, 3):
                g = [r for r in rs if r["seed"] == sd and r["bin"] == b]
                if g:
                    print(f"seed {sd} bin {BINS[b]} beta {sum(r['num'] for r in g)/max(sum(r['den'] for r in g),1e-12):+.3f}")
    ps = [r for r in rows if r.get("kind") == "pos" and ((r["seed"] in HELD) == held)]
    print("-- free rollout of window bodies (closed iso system), mean |pos err| vs iso exact / vs full exact; signed mean x,y vs full")
    for k in (5, 10, 20):
        for b in range(4):
            g = [r for r in ps if r["tick"] == k and r["bin"] == b]
            if not g:
                continue
            n = sum(r["n"] for r in g)
            print(f"err@{k:2d} edge {BINS[b]:>5} n={n:6d} vs_iso {sum(r['pi'] for r in g)/n:.4f} vs_full {sum(r['pf'] for r in g)/n:.4f} "
                  f"signed_vs_full x {sum(r['pfx'] for r in g)/n:+.4f} y {sum(r['pfy'] for r in g)/n:+.4f}")
    es = [r for r in rows if r.get("kind") == "energy" and ((r["seed"] in HELD) == held)]
    for k in (5, 10, 20):
        g = [r for r in es if r["tick"] == k]
        if g:
            dm = [r["E_model"] / r["ke0"] for r in g]
            di = [r["E_iso"] / r["ke0"] for r in g]
            print(f"dE/KE0 @{k}: model {np.mean(dm):+.4f} (se {se(dm):.4f})  iso-exact {np.mean(di):+.4f} (se {se(di):.4f})")


def summ_b(rows, held):
    rs = [r for r in rows if (r["seed"] in HELD) == held]
    print(f"== (b) teacher-forced (tf) vs rollout (ro) residual R=a_exact-a_model, {'held-out' if held else 'dev'}")
    print("fam variant tick | rel_all tf ro ratio | smooth_rel tf ro ratio | seam<8 tf ro | inward_frac tf ro | pos_err")
    for fam in ("cold300", "warm1200"):
        for v in ("single", "g16_fx", "g16_fx_far", "t32_g16_fx", "t32_g16_fx_far"):
            for t in (5, 10, 20, 40):
                g = [r for r in rs if r["fam"] == fam and r["variant"] == v and r["tick"] == t]
                if not g:
                    continue
                m = lambda key, mode: np.mean([r[mode][key] if not isinstance(r[mode][key], dict) else r[mode][key]["rel"] for r in g if (r[mode][key] if not isinstance(r[mode][key], dict) else r[mode][key]["rel"]) is not None])
                a_tf, a_ro = m("all", "tf"), m("all", "ro")
                s_tf, s_ro = m("smooth_rel", "tf"), m("smooth_rel", "ro")
                ratios = [r["ro"]["all"]["rel"] / r["tf"]["all"]["rel"] for r in g]
                sm_ratios = [r["ro"]["smooth_rel"] / r["tf"]["smooth_rel"] for r in g]
                print(f"{fam:8s} {v:15s} {t:3d} | {a_tf:.4f} {a_ro:.4f} {a_ro/a_tf:5.2f} (se {se(ratios):.2f}) | {s_tf:.4f} {s_ro:.4f} {s_ro/s_tf:5.2f} (se {se(sm_ratios):.2f}) | "
                      f"{m('seam<8','tf'):.4f} {m('seam<8','ro'):.4f} | {m('inward_frac','tf'):+.4f} {m('inward_frac','ro'):+.4f} | {np.mean([r['pos_err'] for r in g]):.3f}")


def summ_c(rows, held):
    rs = [r for r in rows if (r["seed"] in HELD) == held]
    print(f"== (c) seam crossing, {'held-out' if held else 'dev'}: per-body specific-energy change de (crossing vs control within 4 of seam) and accel-error jump")
    for fam in ("warm1200", "drift600"):
        for cfg in ("t64", "t32"):
            print(f"-- {fam} {cfg}")
            for run in ("exact", "single", "g16_fx_far" if cfg == "t64" else "t32_g16_fx_far"):
                g = [r for r in rs if r["fam"] == fam and r["cfg"] == cfg and r["run"] == run and r["de_cross"] is not None and r["de_ctrl"] is not None]
                if not g:
                    continue
                ex = [r["de_cross"] - r["de_ctrl"] for r in g]
                line = f"{run:15s} n_cross/scene {np.mean([r['n_cross'] for r in g]):7.0f} de_cross {np.mean([r['de_cross'] for r in g]):+.3e} de_ctrl {np.mean([r['de_ctrl'] for r in g]):+.3e} excess {np.mean(ex):+.3e} (se {se(ex):.1e}) |de|c/|de|k {np.mean([r['absde_cross'] for r in g]):.3e}/{np.mean([r['absde_ctrl'] for r in g]):.3e}"
                if run != "exact":
                    gj = [r for r in g if r.get("jump_cross") is not None and r.get("jump_ctrl") is not None]
                    if gj:
                        jx = [r["jump_cross"] - r["jump_ctrl"] for r in gj]
                        line += f" | jump cross {np.mean([r['jump_cross'] for r in gj]):.4f} ctrl {np.mean([r['jump_ctrl'] for r in gj]):.4f} excess {np.mean(jx):+.4f} (se {se(jx):.4f}) err cross/ctrl {np.mean([r['err_cross'] for r in gj]):.4f}/{np.mean([r['err_ctrl'] for r in gj]):.4f}"
                dE = [r["E"]["40"] / r["ke0"] for r in g]
                line += f" | dE40/KE0 {np.mean(dE):+.4f} (se {se(dE):.4f})"
                print(line)


if __name__ == "__main__":
    d, tag = sys.argv[1], sys.argv[2]
    for held in (True, False):
        try:
            for sc_ in ("sparse", "mid", "dense"):
                summ_a(load(d, "a", tag), held, sc_)
        except FileNotFoundError:
            pass
        try:
            summ_b(load(d, "b", tag), held)
        except FileNotFoundError:
            pass
        try:
            summ_c(load(d, "c", tag), held)
        except FileNotFoundError:
            pass
