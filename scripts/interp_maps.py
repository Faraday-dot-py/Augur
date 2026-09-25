"""H7 interpretability, task 3 and system-level rotation: granularity maps of the learned estimator vs the geometric rule on the flyby
t10000 frame for three kernels, and rotation of the whole state.
Per particle: number of accepted node pairs covering it, mean log2 size of the accepted partner nodes, predicted error (sum over accepted pairs of
lam exp(yhat) m_B |g|, over |a_i|), and (5000 sampled targets) true relative force error.
Usage: PYTHONPATH=. python scripts/interp_maps.py --out results/interp_maps.json"""
import argparse
import json

import numpy as np
import torch

from types import SimpleNamespace

from scripts import adaptive_oracle as ao
from scripts import dual_estimator as de
from scripts import dual_tree as dt
from scripts import est_train
from scripts import interp_kernels as ik
from scripts import kernels

NAMES = ["analytic", "yukawa30", "inv_distance"]
NB = 128


def spearman(x, y):
    def rk(v):
        r = torch.empty(len(v), dtype=torch.float64, device=v.device)
        r[torch.argsort(v)] = torch.arange(len(v), dtype=torch.float64, device=v.device)
        return r
    a, b = rk(x), rk(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def cover(fl, N, ga, vals):
    D = torch.zeros(N + 1, dtype=torch.float64, device=ga.device)
    D.index_add_(0, fl.start[ga], vals)
    D.index_add_(0, fl.start[ga] + fl.count[ga], -vals)
    return torch.cumsum(D, 0)[:N]


def run_est(pos, kern, head, lam=1.0, tol=1e-3):
    kernels.current = kern
    tr = ao.Tree(pos, 18)
    fl = dt.Flat(tr)
    a_c, _, _ = dt.dual_accel(tr, fl, 1.2, 8)
    fs = de.node_scale(fl, a_c)
    col = []
    a_s, m2l, dirn = dt.dual_accel(tr, fl, 1.2, 8, accept_fn=de.make_accept(fl, head, lam, tol, fs), collect=col)
    ga, gb = torch.cat([c[0] for c in col]), torch.cat([c[1] for c in col])
    with torch.no_grad():
        yh = head(de.pair_feats(fl, ga, gb))
    g = kern.f((fl.com[gb] - fl.com[ga]).norm(dim=1))
    pe = lam * torch.exp(yh) * fl.count[gb] * g
    return tr, fl, a_s, ga, gb, pe, (m2l + dirn) / len(pos)


def run_geo(pos, kern, theta):
    kernels.current = kern
    tr = ao.Tree(pos, 18)
    fl = dt.Flat(tr)
    col = []
    a_s, m2l, dirn = dt.dual_accel(tr, fl, theta, 8, collect=col)
    ga, gb = torch.cat([c[0] for c in col]), torch.cat([c[1] for c in col])
    return tr, fl, a_s, ga, gb, (m2l + dirn) / len(pos)


def particle_stats(tr, fl, a_s, ga, gb, pe=None):
    N = len(a_s)
    n = cover(fl, N, ga, torch.ones(len(ga), dtype=torch.float64, device=ga.device))
    ls = cover(fl, N, ga, torch.log2(fl.size[gb]))
    out = {"nacc": n, "msize": ls / n.clamp(min=1)}
    if pe is not None:
        out["prederr"] = cover(fl, N, ga, pe) / a_s.norm(dim=1)
    res = {}
    for k, v in out.items():
        o = torch.empty_like(v)
        o[tr.order] = v
        res[k] = o
    ao_ = torch.empty_like(a_s)
    ao_[tr.order] = a_s
    return res, ao_


def grid_map(xy, v, lo, hi):
    ix = ((xy[:, 0] - lo[0]) / (hi[0] - lo[0]) * NB).long()
    iy = ((xy[:, 1] - lo[1]) / (hi[1] - lo[1]) * NB).long()
    ok = (ix >= 0) & (ix < NB) & (iy >= 0) & (iy < NB)
    k = (ix * NB + iy)[ok]
    cnt = torch.zeros(NB * NB, dtype=torch.float64, device=xy.device).index_add_(0, k, torch.ones(len(k), dtype=torch.float64, device=xy.device))
    if v is None:
        return cnt.view(NB, NB).cpu().numpy()
    s = torch.zeros(NB * NB, dtype=torch.float64, device=xy.device).index_add_(0, k, v[ok].double())
    m = torch.where(cnt >= 3, s / cnt.clamp(min=1), torch.full_like(s, float("nan")))
    return m.view(NB, NB).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/interp_maps.json")
    ap.add_argument("--state", default="flyby_t10000")
    ap.add_argument("--npz", default="results/flyby_100k.npz")
    ap.add_argument("--bodies", type=int, default=100000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cpu" if a.smoke else "cuda")
    pos = ao.load_state(a.state, SimpleNamespace(npz=a.npz, bodies=a.bodies), dev)
    if a.smoke:
        pos = pos[::a.smoke].contiguous()
    N = len(pos)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(4738))[:min(5000, N)].to(dev)
    lo = torch.quantile(pos[::10], 0.001, dim=0)
    hi = torch.quantile(pos[::10], 0.999, dim=0)
    kerns = {n: ik.get(n) for n in NAMES}
    heads = {n: est_train.load(f"checkpoints/interp_est_{n}.pt", dev) for n in NAMES}
    res = {"N": N, "state": a.state, "lo": lo.tolist(), "hi": hi.tolist(), "runs": {}, "matrix": {}, "rotation": {}}
    maps = {"density": grid_map(pos, None, lo, hi)}
    dens = torch.from_numpy(maps["density"]).to(dev).double()
    ix = ((pos[:, 0] - lo[0]) / (hi[0] - lo[0]) * NB).long().clamp(0, NB - 1)
    iy = ((pos[:, 1] - lo[1]) / (hi[1] - lo[1]) * NB).long().clamp(0, NB - 1)
    ld = torch.log(dens[ix, iy])
    cen = pos.mean(0)
    rad = (pos - cen).norm(dim=1)
    redges = torch.logspace(float(torch.log10(torch.quantile(rad[::10], 0.01))), float(torch.log10(torch.quantile(rad[::10], 0.999))), 21, device=dev)
    est_size = {}
    for kn in NAMES:
        kern = kerns[kn]
        exact = kern.exact_accel(pos, idx)
        tr, fl, a_s, ga, gb, pe, cost = run_est(pos, kern, heads[kn])
        st, a_o = particle_stats(tr, fl, a_s, ga, gb, pe)
        terr = (a_o[idx] - exact).norm(dim=1) / exact.norm(dim=1)
        entry = {"est": {"cost": cost, "rel_l2": de.metrics2(a_o[idx], exact)["rel_l2"], "n_pairs": len(ga)}}
        fin = torch.isfinite(terr)
        entry["est"]["n_nonfinite_true_err"] = int((~fin).sum())
        entry["est"]["pred_vs_true_spearman"] = spearman(st["prederr"][idx][fin], terr[fin])
        entry["est"]["pred_err_median"], entry["est"]["true_err_median"] = float(st["prederr"].median()), float(terr[fin].median())
        entry["est"]["corr_logdens"] = {k: spearman(ld, st[k]) for k in ("nacc", "msize", "prederr")}
        est_size[kn] = st["msize"]
        maps[f"{kn}_est_msize"], maps[f"{kn}_est_nacc"], maps[f"{kn}_est_prederr"] = [grid_map(pos, st[k], lo, hi) for k in ("msize", "nacc", "prederr")]
        geos = {}
        for th in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7):
            trg, flg, ag, gag, gbg, cg = run_geo(pos, kern, th)
            geos[th] = cg
            if th == 0.35 or abs(cg - cost) == min(abs(v - cost) for v in geos.values()):
                stg, ago = particle_stats(trg, flg, ag, gag, gbg)
                terr_g = (ago[idx] - exact).norm(dim=1) / exact.norm(dim=1)
                key = "geo035" if th == 0.35 else "geo_matched"
                if th == 0.35:
                    entry["geo035"] = {"cost": cg, "rel_l2": de.metrics2(ago[idx], exact)["rel_l2"], "theta": th, "corr_logdens": {k: spearman(ld, stg[k]) for k in ("nacc", "msize")}}
                    maps[f"{kn}_geo035_msize"], maps[f"{kn}_geo035_nacc"] = [grid_map(pos, stg[k], lo, hi) for k in ("msize", "nacc")]
                    entry["geo035"]["msize_spearman_vs_est"] = spearman(stg["msize"], st["msize"])
                    entry["geo035"]["nacc_spearman_vs_est"] = spearman(stg["nacc"], st["nacc"])
                if abs(cg - cost) == min(abs(v - cost) for v in geos.values()):
                    entry["geo_matched"] = {"cost": cg, "rel_l2": de.metrics2(ago[idx], exact)["rel_l2"], "theta": th, "corr_logdens": {k: spearman(ld, stg[k]) for k in ("nacc", "msize")},
                                            "msize_spearman_vs_est": spearman(stg["msize"], st["msize"]), "nacc_spearman_vs_est": spearman(stg["nacc"], st["nacc"])}
                    maps[f"{kn}_geoM_msize"], maps[f"{kn}_geoM_nacc"] = [grid_map(pos, stg[k], lo, hi) for k in ("msize", "nacc")]
                    gm_size = stg["msize"]
        entry["geo_costs"] = {str(k): v for k, v in geos.items()}
        dq = torch.quantile(ld, torch.linspace(0, 1, 11, device=dev, dtype=torch.float64))
        tab = []
        for i in range(10):
            m = (ld >= dq[i]) & (ld <= dq[i + 1])
            tab.append({"logdens": float(ld[m].mean()), "est_msize": float(st["msize"][m].mean()), "geoM_msize": float(gm_size[m].mean()), "est_nacc": float(st["nacc"][m].mean()),
                        "est_prederr": float(st["prederr"][m].mean())})
        entry["density_decile_table"] = tab
        prof = []
        for i in range(20):
            m = (rad >= redges[i]) & (rad < redges[i + 1])
            if int(m.sum()) > 20:
                prof.append({"r": float(redges[i:i + 2].mean()), "n": int(m.sum()), "logdens": float(ld[m].mean()), "est_msize": float(st["msize"][m].mean()), "geoM_msize": float(gm_size[m].mean()),
                             "est_nacc": float(st["nacc"][m].mean()), "est_prederr": float(st["prederr"][m].mean())})
        entry["radial_profile"] = prof
        res["runs"][kn] = entry
        print(kn, json.dumps({k: entry[k] for k in ("est", "geo035", "geo_matched")}), flush=True)
        json.dump(res, open(a.out, "w"))
        np.savez_compressed(a.out.replace(".json", "_maps.npz"), **maps)
    for kn in NAMES:
        for en in NAMES:
            kern = kerns[kn]
            exact = kern.exact_accel(pos, idx)
            tr, fl, a_s, ga, gb, pe, cost = run_est(pos, kern, heads[en])
            st, a_o = particle_stats(tr, fl, a_s, ga, gb, pe)
            res["matrix"][f"kernel={kn}|est={en}"] = {"cost": cost, "rel_l2": de.metrics2(a_o[idx], exact)["rel_l2"],
                                                       "msize_spearman_vs_own_est": spearman(st["msize"], est_size[kn]), "mean_msize": float(st["msize"].mean())}
        json.dump(res, open(a.out, "w"))
    kern, head = kerns["analytic"], heads["analytic"]
    cenp = pos.mean(0)
    for deg in (0, 15, 30, 45, 90):
        t = torch.tensor(deg * np.pi / 180, dtype=pos.dtype, device=dev)
        R = torch.stack([torch.stack([torch.cos(t), -torch.sin(t)]), torch.stack([torch.sin(t), torch.cos(t)])])
        pr = (pos - cenp) @ R.T + cenp
        exact = kern.exact_accel(pr, idx)
        tr, fl, a_s, ga, gb, pe, cost = run_est(pr, kern, head)
        a_o = torch.empty_like(a_s)
        a_o[tr.order] = a_s
        trg, flg, ag, gag, gbg, cg = run_geo(pr, kern, 0.35)
        ago = torch.empty_like(ag)
        ago[trg.order] = ag
        res["rotation"][str(deg)] = {"est": {"cost": cost, **de.metrics2(a_o[idx], exact)}, "geo035": {"cost": cg, **de.metrics2(ago[idx], exact)}}
        print("rot", deg, json.dumps(res["rotation"][str(deg)]), flush=True)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
