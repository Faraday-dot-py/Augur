"""Orbiting-body analysis for scatter_disc_run.py output (pass/kill criterion in docs/debugging/experiment-log.md, solar-10M entry).

Clumps (candidate bodies) are density peaks in the star-centred subsample at each check tick; the first check tick with >= --need clumps is the formation tick t_f. Members of each clump are tracked by id for --horizon ticks; the clump centre of mass is tested against the central body (median of core particles): bound (E<0 with mu = enclosed mass) at every snapshot, window-averaged (--win ticks) semi-major-axis drift < --drift, coherent (>= --coh of member mass within 1.5 of the centre of mass) at every snapshot.

Usage: python scripts/analyze_disc.py results/disc/<tag> [--horizon 1000]
"""
import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

EPS = 0.5


def find_clumps(rel, w, args):
    """w: per-particle weights (K,)."""
    cell = 0.25
    hw = 20.0
    nb = int(2 * hw / cell)
    H, _, _ = np.histogram2d(rel[:, 0], rel[:, 1], bins=nb, range=[[-hw, hw], [-hw, hw]], weights=w)
    Hs = gaussian_filter(H, 2.0)
    c = (np.arange(nb) + 0.5) * cell - hw
    R = np.hypot(c[:, None], c[None, :])
    rb = np.minimum((R).astype(int), 40)
    ring = np.bincount(rb.ravel(), Hs.ravel(), minlength=41) / np.maximum(np.bincount(rb.ravel(), minlength=41), 1)
    over = Hs / np.maximum(ring[rb], 1e-12)
    pk = (Hs == maximum_filter(Hs, size=9)) & (over >= args.over) & (R >= args.rmin) & (R <= args.rmax)
    cand = []
    for ix, iy in zip(*np.nonzero(pk)):
        ctr = np.array([c[ix], c[iy]])
        d = np.linalg.norm(rel - ctr, axis=1)
        mem = np.nonzero(d < args.aperture)[0]
        m = float(w[mem].sum())
        if m >= args.min_mass:
            cand.append((m, ctr, mem))
    cand.sort(key=lambda x: -x[0])
    keep = []
    for m, ctr, mem in cand:
        if all(np.linalg.norm(ctr - k[1]) > 2.0 for k in keep):
            keep.append((m, ctr, mem))
    return keep


def centre(P, V, core, j):
    return np.median(P[j, core], axis=0), np.median(V[j, core], axis=0)


def track(P, V, mem, core, w, i0, i1):
    out = []
    for j in range(i0, i1 + 1):
        star_p, star_v = centre(P, V, core, j)
        wm = w[mem] / w[mem].sum()
        com = (P[j, mem] * wm[:, None]).sum(0)
        cv = (V[j, mem] * wm[:, None]).sum(0)
        rel, vr = com - star_p, cv - star_v
        r = np.linalg.norm(rel)
        rd = np.linalg.norm(P[j] - star_p, axis=1)
        menc = float(w[rd < r].sum())
        E = 0.5 * (vr ** 2).sum() - menc / np.sqrt(r ** 2 + EPS ** 2)
        L = rel[0] * vr[1] - rel[1] * vr[0]
        a = -menc / (2 * E) if E < 0 else np.inf
        e = np.sqrt(max(0.0, 1 + 2 * E * L ** 2 / menc ** 2))
        dm = np.linalg.norm(P[j, mem] - com, axis=1)
        out.append(dict(j=j, r=r, E=E, a=a, e=e, mu=menc, com=com, med_dist=float(np.median(dm)), frac_in=float(w[mem][dm < 1.5].sum() / w[mem].sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--need", type=int, default=3)
    ap.add_argument("--min-mass", type=float, default=0.1)
    ap.add_argument("--over", type=float, default=3.0)
    ap.add_argument("--aperture", type=float, default=1.0)
    ap.add_argument("--rmin", type=float, default=2.0)
    ap.add_argument("--rmax", type=float, default=18.0)
    ap.add_argument("--drift", type=float, default=0.25)
    ap.add_argument("--coh", type=float, default=0.5)
    ap.add_argument("--win", type=int, default=100, help="semi-major axis is averaged over windows of this many ticks before the drift test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    meta = json.load(open(f"{args.dir}/meta.json"))
    every = meta["snap_every"]
    P = np.load(f"{args.dir}/snaps_pos.npy", mmap_mode="r")
    V = np.load(f"{args.dir}/snaps_vel.npy", mmap_mode="r")
    diag = json.load(open(f"{args.dir}/diag.json"))
    last = diag[-1]["tick"]
    last_i = min(P.shape[0] - 1, last // every)
    K = P.shape[1]
    sub = np.load(f"{args.dir}/sub_idx.npy")
    core = np.nonzero(sub < meta["ncore"])[0]
    w = np.load(f"{args.dir}/sub_wt.npy")
    P = np.asarray(P[: last_i + 1], dtype=np.float64)
    V = np.asarray(V[: last_i + 1], dtype=np.float64)
    step = max(1, args.check_every // every)
    hz = args.horizon // every
    scan = []
    result = None
    for i in range(0, last_i + 1, step):
        rel = P[i] - centre(P, V, core, i)[0]
        cl = find_clumps(rel, w, args)
        row = dict(tick=i * every, n_clumps=len(cl), masses=[round(m, 3) for m, _, _ in cl[:8]])
        if len(cl) >= args.need:
            i1 = min(last_i, i + hz)
            trs = [track(P, V, mem, core, w, i, i1) for _, _, mem in cl]
            rows = []
            for (m, ctr, mem), tr in zip(cl, trs):
                a0 = tr[0]["a"]
                bound = all(x["E"] < 0 for x in tr)
                av = np.array([x["a"] for x in tr])
                nw = max(1, args.win // every)
                nw = min(nw, len(av))
                wa = np.array([av[k:k + nw].mean() for k in range(0, len(av) - nw + 1, nw)])
                a0 = wa[0]
                drift = float(np.abs(wa - a0).max() / a0) if bound else np.inf
                coh = min(x["frac_in"] for x in tr)
                rows.append(dict(mass=m, a0=a0, a_end=float(wa[-1]), e0=tr[0]["e"], e_end=tr[-1]["e"], r0=tr[0]["r"], r_end=tr[-1]["r"], bound_all=bool(bound), max_drift=float(drift),
                                 min_frac_in=coh, passes=bool(bound and drift < args.drift and coh >= args.coh), com_end=tr[-1]["com"].tolist()))
            sep_ok = []
            for a_ in range(len(trs)):
                ok = True
                for b_ in range(len(trs)):
                    if a_ != b_ and min(np.linalg.norm(x["com"] - y["com"]) for x, y in zip(trs[a_], trs[b_])) < 1.5:
                        ok = False
                sep_ok.append(ok)
            for r_, s_ in zip(rows, sep_ok):
                r_["distinct"] = s_
                r_["passes"] = bool(r_["passes"] and s_)
            row.update(track_end_tick=i1 * every, full_horizon=bool(i1 - i == hz), n_pass=sum(r_["passes"] for r_ in rows), bodies=rows)
            if result is None:
                result = row
        scan.append(row)
    res = dict(formation=result, scan=scan, n_pass=None if result is None else result["n_pass"], last_tick=last_i * every, params=vars(args))
    jj = min(last_i, hz)
    rel = P[jj] - centre(P, V, core, jj)[0]
    res["central_mass_within_2_at_horizon"] = float(w[np.linalg.norm(rel, axis=1) < 2.0].sum())
    res["central_mass_within_2_at_0"] = float(w[np.linalg.norm(P[0] - centre(P, V, core, 0)[0], axis=1) < 2.0].sum())
    res["centre_drift"] = float(np.linalg.norm(centre(P, V, core, last_i)[0] - centre(P, V, core, 0)[0]))
    json.dump(res, open(args.out or f"{args.dir}/analysis.json", "w"), indent=1, default=float)
    print("tick  n_clumps masses")
    for r_ in scan:
        print(r_["tick"], r_["n_clumps"], r_["masses"], r_.get("n_pass", ""))
    if result:
        print(f"formation tick {result['tick']} horizon end {result['track_end_tick']} full_horizon {result['full_horizon']} n_pass {result['n_pass']}")
        for b in result["bodies"]:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if k != "com_end"})
    else:
        print("NO FORMATION: never >= need clumps")


if __name__ == "__main__":
    main()
