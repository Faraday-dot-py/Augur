"""Attribution, pre-registered lines and figures from the per-group files of scatter_three_part_exact_swap_drift_attribution.py.

  PYTHONPATH=. python scripts/scatter_three_part_exact_swap_drift_attribution_analyze.py --in-dir results/scatter_field_three_part_exact_swap_drift_attribution
Writes attribution.json, figure-err-by-arm.png, figure-part-shares.png, figure-force-error-by-part.png next to the inputs.
"""
import argparse
import glob
import itertools
import json
import math
import os

import numpy as np

MODELS = {"exp-e-frozen": "Exp E frozen (grid 128)", "width8-single-law-control": "width-8 single-law control"}
SETS = ("heldout", "fresh")
PARTS = ("kernel", "U-Net", "pair term")
COL = {"kernel": "#0072B2", "U-Net": "#D55E00", "pair term": "#009E73", "total": "#222222"}
KS = (20, 50, 100)
LINE_DOMINANT, LINE_FAIL, TIE_PP = 0.5, 0.25, 0.05
FLOOR_GUARD = 0.25
TICKS_SHOWN = (1, 5, 10, 20, 30, 50, 75, 100, 125, 150)


def arm_name(a):
    ex = [p for i, p in enumerate(PARTS) if (a >> i) & 1]
    return "all learned" if not ex else "all exact" if len(ex) == 3 else "exact " + " + ".join(ex)


def load(in_dir, model, sset, fam=None):
    errs, recs, tfs, refs, units = [], [], [], [], []
    for f in sorted(glob.glob(f"{in_dir}/{model}_{sset}_*.npz")):
        d = np.load(f)
        grp = json.loads(str(d["group"]))
        if fam and grp[1] != fam:
            continue
        errs.append(d["err"])
        recs.append(d["rec"])
        tfs.append(d["tf"])
        refs.append(d["ref"])
        units.append(json.loads(str(d["unit"])))
    if not errs:
        return None
    return dict(err=np.concatenate(errs, 1), rec=np.concatenate(recs, 1), tf=np.concatenate(tfs, 1), ref=np.concatenate(refs, 0), units=units)


def shapley(v):
    """v[S] for S a bitmask 0..7 (error reduction when parts in S are exact, over time axis). Returns (3, T)."""
    out = np.zeros((3,) + np.shape(v[0]))
    for i in range(3):
        others = [j for j in range(3) if j != i]
        for r in range(3):
            w = math.factorial(r) * math.factorial(2 - r) / 6.0
            for sub in itertools.combinations(others, r):
                s = sum(1 << j for j in sub)
                out[i] += w * (v[s | (1 << i)] - v[s])
    return out


def attribution(err):
    """err (8, S, T+1) -> dict of curves from scene-mean error."""
    e = err.mean(1)
    v = {s: e[0] - e[s] for s in range(8)}
    phi = shapley(v)
    tot = e[0] - e[7]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(np.abs(tot) > 1e-12, phi / tot, np.nan)
    single = np.stack([e[0] - e[1 << i] for i in range(3)])
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = single / e[0]
        gain_floor = single / (e[0] - e[7])
    return dict(e=e, phi=phi, tot=tot, share=share, single=single, gain=gain, gain_floor=gain_floor)


def verdict_line(gain):
    order = np.argsort(-gain)
    top, second = int(order[0]), int(order[1])
    g = [float(x) for x in gain]
    if g[top] > LINE_DOMINANT or math.isclose(g[top], LINE_DOMINANT):
        if g[top] - g[second] < TIE_PP and g[second] >= LINE_DOMINANT:
            return dict(outcome="tie", parts=[PARTS[top], PARTS[second]], gains=g)
        return dict(outcome="dominant", part=PARTS[top], gains=g)
    if g[top] <= LINE_FAIL:
        return dict(outcome="fails: no dominant source", gains=g)
    return dict(outcome="intermediate: no part reaches the dominance line, one exceeds the failure line", top=PARTS[top], gains=g)


def prereg(a, T):
    out = {}
    for k in KS:
        if k > T:
            continue
        floor_ratio = float(a["e"][7][k] / a["e"][0][k])
        use_floor = floor_ratio > FLOOR_GUARD
        gain = a["gain_floor"][:, k] if use_floor else a["gain"][:, k]
        row = verdict_line(gain)
        row.update(err_all_learned=float(a["e"][0][k]), err_all_exact_floor=float(a["e"][7][k]), floor_over_all_learned=floor_ratio,
                   floor_adjusted_used=use_floor, gains_unadjusted=[float(x) for x in a["gain"][:, k]],
                   gains_floor_adjusted=[float(x) for x in a["gain_floor"][:, k]])
        out[str(k)] = row
    return out


def boot_ci(err, k, reps=1000, seed=4738):
    rng = np.random.default_rng(seed)
    S = err.shape[1]
    res = []
    for _ in range(reps):
        i = rng.integers(0, S, S)
        e = err[:, i, k].mean(1)
        res.append([(e[0] - e[1 << p]) / e[0] for p in range(3)])
    res = np.array(res)
    return {PARTS[p]: [float(np.percentile(res[:, p], 2.5)), float(np.percentile(res[:, p], 97.5))] for p in range(3)}


def dominance(a, T):
    sh = a["share"]
    dom = [None] + [int(np.nanargmax(sh[:, t])) if np.isfinite(sh[:, t]).all() else None for t in range(1, T + 1)]
    changes, run_start = [], 1
    cur = dom[1]
    for t in range(2, T + 1):
        if dom[t] is not None and dom[t] != cur:
            changes.append(dict(tick=t, from_part=PARTS[cur] if cur is not None else None, to_part=PARTS[dom[t]], sustained_ticks=0))
            cur, run_start = dom[t], t
    for c in changes:
        t = c["tick"]
        n = 0
        while t + n <= T and dom[t + n] == dom[t]:
            n += 1
        c["sustained_ticks"] = n
    return dom, changes


def force_table(rec_tf, rec_own):
    def rel(r, t):
        s = r[:, t].sum(0)
        return {"kernel": float(math.sqrt(s[0] / s[4])), "U-Net": float(math.sqrt(s[1] / s[4])), "pair term": float(math.sqrt(s[2] / s[4])),
                "total": float(math.sqrt(s[3] / s[4]))}
    T = rec_tf.shape[1] - 1
    return {str(t): {"teacher_forced": rel(rec_tf, t), "own_trajectory": rel(rec_own, t)} for t in TICKS_SHOWN if t <= T}


def analyze(in_dir):
    res = {}
    for model in MODELS:
        for sset in SETS:
            for fam in (None, "expB", "cluster"):
                d = load(in_dir, model, sset, fam)
                if d is None:
                    continue
                T = d["err"].shape[2] - 1
                a = attribution(d["err"])
                dom, changes = dominance(a, T)
                key = f"{model}/{sset}/{fam or 'all-families'}"
                shares = {str(t): {PARTS[i]: float(a["share"][i, t]) for i in range(3)} for t in TICKS_SHOWN if t <= T}
                single_sum = a["single"].sum(0)
                interaction = {str(k): dict(sum_of_single_gains=float(single_sum[k]), all_exact_gain=float(a["tot"][k]),
                                            ratio=float(single_sum[k] / a["tot"][k]) if a["tot"][k] else None) for k in KS if k <= T}
                res[key] = dict(scenes=int(d["err"].shape[1]), ticks=T,
                                err_by_arm={arm_name(s): {str(k): float(a["e"][s][k]) for k in (5, 10, 20, 50, 100, 150) if k <= T} for s in range(8)},
                                reference_exact_force_verlet_err={str(k): float(d["ref"].mean(0)[k]) for k in KS if k <= T},
                                prereg_lines=prereg(a, T), single_swap_gain_ci95={str(k): boot_ci(d["err"], k) for k in KS if k <= T},
                                shapley_share_at_ticks=shares, dominant_part_by_tick={str(t): (PARTS[x] if x is not None else None) for t, x in enumerate(dom)},
                                dominance_changes=changes, interaction=interaction,
                                force_rel_by_part=force_table(d["tf"][0], d["rec"][0]),
                                unit_pass_all_groups=all(u["pass"] for u in d["units"]),
                                unit_summary=dict(
                                    kernel_exact_over_force_relrms=[u["kernel_exact_over_force_relrms"] for u in d["units"]],
                                    unet_exact_over_force_relrms=[u["unet_exact_over_force_relrms"] for u in d["units"]],
                                    pair_exact_over_force_relrms=[u["pair_exact_over_force_relrms"] for u in d["units"]],
                                    parts_sum_vs_model_force_relmax=max(u["parts_sum_vs_model_force_relmax"] for u in d["units"]),
                                    all_exact_teacher_forced_force_rel=max(u["all_exact_teacher_forced_force_rel"] for u in d["units"])))
                if fam is None:
                    res[key]["_raw"] = (a, dom, changes, d)
    return res


def fig_err(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    cmap = plt.get_cmap("tab10")
    sty = ["-", "-", "-", "--", "-", "--", "--", ":"]
    for r, model in enumerate(MODELS):
        for c, sset in enumerate(SETS):
            ax = axs[r, c]
            raw = res.get(f"{model}/{sset}/all-families", {}).get("_raw")
            if raw is None:
                continue
            a = raw[0]
            for s in range(8):
                ax.plot(np.arange(len(a["e"][s])), np.maximum(a["e"][s], 1e-6), sty[s], color=cmap(s) if s < 7 else "black", lw=2 if s in (0, 7) else 1.5, label=arm_name(s))
            ax.set_yscale("log")
            ax.set_title(f"{MODELS[model]}, {'held-out' if sset == 'heldout' else 'fresh'} scenes ({raw[3]['err'].shape[1]} scenes)")
            ax.set_ylabel("mean body position error (length units)")
            ax.set_xlabel("rollout tick (0.1 time units each)")
            ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_share(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for r, model in enumerate(MODELS):
        for c, sset in enumerate(SETS):
            ax = axs[r, c]
            raw = res.get(f"{model}/{sset}/all-families", {}).get("_raw")
            if raw is None:
                continue
            a, dom, changes, d = raw
            T = a["e"].shape[1] - 1
            t = np.arange(1, T + 1)
            for i, p in enumerate(PARTS):
                ax.plot(t, a["share"][i, 1:], color=COL[p], lw=2, label=p)
            lo = max(min(-0.1, np.nanmin(a["share"][:, 1:]) - 0.05), -1.5)
            hi = min(max(1.1, np.nanmax(a["share"][:, 1:]) + 0.05), 2.5)
            ax.set_ylim(lo, hi)
            strip = lo + 0.02 * (hi - lo)
            for tt in t:
                if dom[tt] is not None:
                    ax.plot([tt], [strip], "s", color=COL[PARTS[dom[tt]]], ms=4)
            for ch in changes:
                if ch["sustained_ticks"] >= 5:
                    ax.axvline(ch["tick"], color="gray", ls="--", lw=1)
                    ax.text(ch["tick"] + 1, hi - 0.08 * (hi - lo), f"tick {ch['tick']}: {ch['to_part']}", fontsize=10, rotation=90, va="top")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{MODELS[model]}, {'held-out' if sset == 'heldout' else 'fresh'} (squares: dominant part)")
            ax.set_ylabel("share of drift error removed (average over swap orders)")
            ax.set_xlabel("rollout tick (0.1 time units each)")
            ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=11, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_force(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for c, model in enumerate(MODELS):
        raw = res.get(f"{model}/heldout/all-families", {}).get("_raw")
        if raw is None:
            continue
        d = raw[3]
        ax = axs[c]
        T = d["err"].shape[2] - 1
        t = np.arange(T + 1)
        for k, nm in enumerate(("kernel", "U-Net", "pair term", "total")):
            for rec, ls, lab in ((d["tf"][0], "-", "teacher-forced"), (d["rec"][0], "--", "own trajectory")):
                s = rec.sum(0)
                ax.plot(t, np.sqrt(s[:, k] / s[:, 4]), ls, color=COL[nm], lw=2, label=f"{nm}, {lab}")
        ax.set_yscale("log")
        ax.set_title(f"{MODELS[model]}, held-out scenes")
        ax.set_xlabel("rollout tick (0.1 time units each)")
        ax.set_ylabel("force error of the part, relative to exact force (rms)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    a = ap.parse_args()
    res = analyze(a.in_dir)
    fig_err(res, f"{a.in_dir}/figure-err-by-arm.png")
    fig_share(res, f"{a.in_dir}/figure-part-shares.png")
    fig_force(res, f"{a.in_dir}/figure-force-error-by-part.png")
    for v in res.values():
        v.pop("_raw", None)
    json.dump(res, open(f"{a.in_dir}/attribution.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v["scenes"], "scenes")
        for kk, row in v["prereg_lines"].items():
            print("  err@" + kk, row["outcome"], row.get("part", row.get("parts", row.get("top"))), [round(x, 3) for x in row["gains_unadjusted"]], "floor/all", round(row["floor_over_all_learned"], 3))
        print("  dominance changes", [(c["tick"], c["to_part"], c["sustained_ticks"]) for c in v["dominance_changes"]][:8])


if __name__ == "__main__":
    main()
