"""Per-particle error-distribution analysis of rollouts from scripts/loss_error_roll.py. Definitions in docs/debugging/loss-analysis-2026-10-26.md.
Usage: python3 scripts/loss_error_analyze.py results/loss_analysis/roll_n300.npz --out results/loss_analysis/stats_n300.json"""
import argparse
import json

import numpy as np

KS = [5, 20, 50, 100]
K_NB = 16
R_DENS = 3.0


def kurt(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2 - 3)


def neighbours(p):
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return np.argsort(d, 1)[:, :K_NB], (d < R_DENS).sum(1) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.npz)
    arms = [k[:-4] for k in d.files if k.endswith("_pos") and k != "truth_pos"]
    S = d["truth_pos"].shape[0]
    res = {}
    for arm in arms:
        res[arm] = {}
        for k in KS:
            pool = dict(e2=[], e=[], dens=[], mass=[], coh2=[], res2=[], rad=[], tan=[], vel=[], dke=[])
            per_scene_top1 = []
            for s in range(S):
                pt, vt = d["truth_pos"][s, k], d["truth_vel"][s, k]
                pm, vm = d[f"{arm}_pos"][s, k], d[f"{arm}_vel"][s, k]
                if not (np.isfinite(pm).all() and np.isfinite(vm).all()):
                    continue
                ev = pm - pt
                e = np.linalg.norm(ev, axis=1)
                nb, dens = neighbours(pt)
                coh = ev[nb].mean(1)
                resid = ev - coh
                cen = pt[nb].mean(1)
                rv = pt - cen
                rn = np.linalg.norm(rv, axis=1) + 1e-9
                rh = rv / rn[:, None]
                th = np.stack([-rh[:, 1], rh[:, 0]], 1)
                vh = vt / (np.linalg.norm(vt, axis=1)[:, None] + 1e-9)
                m = d["mass"][s]
                pool["e2"].append(e ** 2), pool["e"].append(e), pool["dens"].append(dens), pool["mass"].append(m)
                pool["coh2"].append((coh ** 2).sum(1)), pool["res2"].append((resid ** 2).sum(1))
                pool["rad"].append((ev * rh).sum(1)), pool["tan"].append((ev * th).sum(1)), pool["vel"].append((ev * vh).sum(1))
                pool["dke"].append(0.5 * m * ((vm ** 2).sum(1) - (vt ** 2).sum(1)))
                n1 = max(1, int(round(0.01 * len(e))))
                per_scene_top1.append(float(np.sort(e ** 2)[-n1:].sum() / (e ** 2).sum()))
            if not pool["e2"]:
                res[arm][str(k)] = "non-finite"
                continue
            c = {q: np.concatenate(v) for q, v in pool.items()}
            e2 = c["e2"]
            srt = np.sort(e2)[::-1]
            tot = e2.sum()
            top = lambda f: float(srt[:max(1, int(round(f * len(srt))))].sum() / tot)
            qd = np.quantile(c["dens"], [0.5, 0.9])
            bins = {"low50": c["dens"] <= qd[0], "mid40": (c["dens"] > qd[0]) & (c["dens"] <= qd[1]), "dense10": c["dens"] > qd[1]}
            qm = np.quantile(c["mass"], [1 / 3, 2 / 3])
            mb = {"light": c["mass"] <= qm[0], "medium": (c["mass"] > qm[0]) & (c["mass"] <= qm[1]), "heavy": c["mass"] > qm[1]}
            if qm[0] == qm[-1]:
                mb = {}
            r = dict(err_mean=float(c["e"].mean()), err_median=float(np.median(c["e"])), err_p99=float(np.quantile(c["e"], 0.99)),
                     top1_share=top(0.01), top5_share=top(0.05), top1_share_per_scene_mean=float(np.mean(per_scene_top1)),
                     kurtosis=kurt(c["e"]), loss_pos_scalar=float(e2.mean() / 2),
                     coherent_share=float(c["coh2"].sum() / (c["coh2"].sum() + c["res2"].sum())),
                     dir_rms=dict(radial=float(np.sqrt((c["rad"] ** 2).mean())), tangential=float(np.sqrt((c["tan"] ** 2).mean())), along_vel=float(np.sqrt((c["vel"] ** 2).mean()))),
                     dir_mean_signed=dict(radial=float(c["rad"].mean()), along_vel=float(c["vel"].mean())),
                     by_density={n: dict(frac_particles=float(b.mean()), share_sq_err=float(e2[b].sum() / tot), mean_err=float(c["e"][b].mean()),
                                         mean_signed_dKE=float(c["dke"][b].mean()), coherent_share=float(c["coh2"][b].sum() / (c["coh2"][b].sum() + c["res2"][b].sum()))) for n, b in bins.items()},
                     by_mass={n: dict(share_sq_err=float(e2[b].sum() / tot), mean_err=float(c["e"][b].mean())) for n, b in mb.items()},
                     top1_in_dense10=float(np.isin(np.where(e2 >= srt[max(0, int(round(0.01 * len(srt))) - 1)])[0], np.where(bins["dense10"])[0]).mean()))
            res[arm][str(k)] = r
    json.dump(res, open(a.out, "w"), indent=1)
    for arm, v in res.items():
        for k, r in v.items():
            if isinstance(r, str):
                print(arm, k, r)
                continue
            print(f"{arm:12s} k={k:>3s} err {r['err_mean']:.3f} med {r['err_median']:.3f} top1% {r['top1_share']:.2f} top5% {r['top5_share']:.2f} kurt {r['kurtosis']:.1f} coh {r['coherent_share']:.2f} "
                  f"dense10 sq-share {r['by_density']['dense10']['share_sq_err']:.2f} low50 {r['by_density']['low50']['share_sq_err']:.2f} rad/tan/vel rms {r['dir_rms']['radial']:.3f}/{r['dir_rms']['tangential']:.3f}/{r['dir_rms']['along_vel']:.3f} loss_pos {r['loss_pos_scalar']:.4f}")


if __name__ == "__main__":
    main()
