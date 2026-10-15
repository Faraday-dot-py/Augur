"""Exact-force integrator budget for docs/debugging/substep-theory.md (CPU, fp64, no model).

  PYTHONPATH=. python scripts/scatter_substep_theory.py --out results/substep_theory.json
Part A: per-body stats at truth states (nn, speeds, |a|, jerk, timescales) and the one-step error of 1 Verlet step/tick
(and KDK 2 substeps) vs the 4-substep truth, by nn bin; capture curves (promoted fraction f vs share of one-step error);
modified-force consistency (force bias needed to fix position vs velocity error).
Part B: free rollouts with exact force: Verlet-1, uniform KDK 2/4, per-body adaptive substeps (promoted set at tick start,
inactive bodies on predicted drift paths), with full force or near-pair-only recompute (far part frozen / extrapolated).
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf

DEV = torch.device("cpu")
DT, EPS, TR = 0.1, 0.5, 100
KS = (5, 20, 50, 100)
NNB = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9)
FS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
torch.set_default_dtype(torch.float64)


def pairs(p, m, rc=None):
    d = p[None] - p[:, None]
    r2 = (d ** 2).sum(-1)
    inv = (r2 + EPS ** 2) ** -1.5
    inv.fill_diagonal_(0.0)
    if rc is not None:
        inv = inv * (r2 < rc ** 2)
    return d, r2, inv * m[None]


def acc(p, m, rc=None):
    d, _, w = pairs(p, m, rc)
    return (d * w[..., None]).sum(1)


def jerk(p, v, m):
    d, r2, w = pairs(p, m)
    dv = v[None] - v[:, None]
    s2 = r2 + EPS ** 2
    dd = (d * dv).sum(-1)
    return (w[..., None] * (dv - 3 * (dd / s2)[..., None] * d)).sum(1)


def nn_info(p, v):
    d = p[None] - p[:, None]
    r = d.norm(dim=-1) + torch.eye(len(p)) * 1e9
    nn, j = r.min(1)
    vrel = (v[j] - v).norm(dim=-1)
    return nn, vrel


def kdk(p, v, m, M):
    h = DT / M
    a = acc(p, m)
    for _ in range(M):
        v = v + 0.5 * h * a
        p = p + h * v
        a = acc(p, m)
        v = v + 0.5 * h * a
    return p, v


def make_scenes(fam, num, seed):
    if fam == "expB":
        data = gs.make_dataset(num, (10, 100), 1, seed, scale=True, device=DEV, dt=DT, eps=EPS)
    else:
        data = sf.make_cluster_dataset(num, 1, seed, DEV, dt=DT, eps=EPS)
    out = []
    for d in data:
        p = torch.tensor(d[0][0] - sf.CENTER)
        v = torch.tensor(d[1][0])
        m = torch.ones(len(p))
        P, V = [p], [v]
        for _ in range(TR):
            p, v = kdk(p, v, m, 4)
            P.append(p)
            V.append(v)
        out.append(dict(P=torch.stack(P), V=torch.stack(V), m=m))
    return out


def energy(p, v, m):
    d = p[None] - p[:, None]
    r = torch.sqrt((d ** 2).sum(-1) + EPS ** 2)
    iu = torch.triu_indices(len(p), len(p), 1)
    return float(0.5 * (m * (v ** 2).sum(-1)).sum() - (m[iu[0]] * m[iu[1]] / r[iu[0], iu[1]]).sum())


def part_a(scenes):
    cols = {k: [] for k in ("nn", "vrel", "speed", "a", "j", "e1", "e2", "ev1", "bx", "bv", "cos")}
    for sc in scenes:
        m = sc["m"]
        for t in range(TR):
            p, v = sc["P"][t], sc["V"][t]
            pt, vt = sc["P"][t + 1], sc["V"][t + 1]
            nn, vrel = nn_info(p, v)
            a = acc(p, m)
            j = jerk(p, v, m)
            p1, v1 = kdk(p, v, m, 1)
            p2, _ = kdk(p, v, m, 2)
            dx, dvv = pt - p1, vt - v1
            bx, bv = 2 * dx / DT ** 2, dvv / DT
            cols["nn"].append(nn)
            cols["vrel"].append(vrel)
            cols["speed"].append(v.norm(dim=-1))
            cols["a"].append(a.norm(dim=-1))
            cols["j"].append(j.norm(dim=-1))
            cols["e1"].append(dx.norm(dim=-1))
            cols["e2"].append((pt - p2).norm(dim=-1))
            cols["ev1"].append(dvv.norm(dim=-1))
            cols["bx"].append(bx.norm(dim=-1))
            cols["bv"].append(bv.norm(dim=-1))
            cols["cos"].append((bx * bv).sum(-1) / (bx.norm(dim=-1) * bv.norm(dim=-1)).clamp_min(1e-30))
    c = {k: torch.cat(v) for k, v in cols.items()}
    s = torch.sqrt(c["nn"] ** 2 + EPS ** 2)
    c["tau_enc"] = s / c["vrel"].clamp_min(1e-9)
    c["tau_ff"] = torch.sqrt(s ** 3 / 2.0)
    c["tau_j"] = c["a"] / c["j"].clamp_min(1e-12)
    c["jpred"] = DT ** 3 * c["j"] / 6
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    out = {"n_body_ticks": int(len(c["nn"]))}
    out["quantiles(.05,.25,.5,.75,.95,.99)"] = {k: [float(x) for x in torch.quantile(c[k], q)] for k in
                                               ("nn", "vrel", "speed", "a", "j", "tau_enc", "tau_ff", "tau_j", "e1", "e2", "ev1")}
    out["frac_nn_lt"] = {str(r): float((c["nn"] < r).double().mean()) for r in (0.25, 0.5, 1.0, 2.0)}
    out["frac_tau_lt"] = {f"{k}<{n}dt": float((c[k] < n * DT).double().mean()) for k in ("tau_enc", "tau_ff", "tau_j") for n in (1, 2, 4, 10)}
    e1s, e1sum, ev1s = (c["e1"] ** 2).sum(), c["e1"].sum(), (c["ev1"] ** 2).sum()
    b = torch.bucketize(c["nn"], torch.tensor(NNB[1:-1]))
    bins = []
    for i in range(len(NNB) - 1):
        k = b == i
        if not k.any():
            continue
        bins.append(dict(nn=f"{NNB[i]}-{NNB[i + 1]}", frac=float(k.double().mean()),
                         e1_mean=float(c["e1"][k].mean()), e1_share_sq=float((c["e1"][k] ** 2).sum() / e1s),
                         e1_share=float(c["e1"][k].sum() / e1sum), ev1_share_sq=float((c["ev1"][k] ** 2).sum() / ev1s),
                         e2_over_e1=float(c["e2"][k].mean() / c["e1"][k].mean()), ev1_mean=float(c["ev1"][k].mean()),
                         bx_rel_med=float((c["bx"][k] / c["a"][k].clamp_min(1e-12)).median()),
                         bv_rel_med=float((c["bv"][k] / c["a"][k].clamp_min(1e-12)).median()),
                         bx_over_bv_med=float((c["bx"][k] / c["bv"][k].clamp_min(1e-30)).median()),
                         cos_bx_bv_med=float(c["cos"][k].median()),
                         tau_enc_med=float(c["tau_enc"][k].median()), tau_ff_med=float(c["tau_ff"][k].median()),
                         vrel_med=float(c["vrel"][k].median()), a_med=float(c["a"][k].median()),
                         jpred_over_e1_med=float((c["jpred"][k] / c["e1"][k].clamp_min(1e-30)).median())))
    out["bins"] = bins
    scores = {"nn(asc)": -c["nn"], "jerk_err(desc)": c["jpred"], "|a|(desc)": c["a"], "1/tau_enc(desc)": -c["tau_enc"],
              "oracle_e1(desc)": c["e1"]}
    cap = {}
    for name, sc in scores.items():
        o = torch.argsort(sc, descending=True)
        e2, e = (c["e1"][o] ** 2).cumsum(0) / e1s, c["e1"][o].cumsum(0) / e1sum
        n = len(o)
        cap[name] = {str(f): [float(e2[max(0, int(f * n) - 1)]), float(e[max(0, int(f * n) - 1)])] for f in FS}
    out["capture_[share_err2,share_err]"] = cap
    r1, r2 = torch.argsort(torch.argsort(c["jpred"])).double(), torch.argsort(torch.argsort(c["e1"])).double()
    out["spearman_jpred_e1"] = float(torch.corrcoef(torch.stack([r1, r2]))[0, 1])
    return out


def adaptive_tick(p, v, a0, m, act, M, mode, rc, far_prev):
    h = DT / M
    pa, va, aa = p.clone(), v.clone(), a0.clone()
    na = int(act.sum())
    if mode != "full" and na:
        near0 = acc(p, m, rc)
        far0 = a0 - near0
        dfar = (far0 - far_prev) if (mode == "extrap" and far_prev is not None) else torch.zeros_like(far0)
    for s in range(1, M + 1):
        tau = s * h
        va[act] += 0.5 * h * aa[act]
        pa[act] += h * va[act]
        if s < M and na:
            pos = p + tau * v + 0.5 * tau ** 2 * a0
            pos[act] = pa[act]
            if mode == "full":
                aa[act] = acc(pos, m)[act]
            else:
                aa[act] = (far0 + dfar * tau / DT + acc(pos, m, rc))[act]
            va[act] += 0.5 * h * aa[act]
    pn = p + DT * v + 0.5 * DT ** 2 * a0
    pn[act] = pa[act]
    an = acc(pn, m)
    vn = v + 0.5 * DT * (a0 + an)
    vn[act] = va[act] + 0.5 * h * an[act]
    far_n = (an - acc(pn, m, rc)) if mode != "full" else None
    return pn, vn, an, far_n


def criterion(name, p, v, m):
    if name == "none":
        return torch.zeros(len(p), dtype=torch.bool)
    if name == "all":
        return torch.ones(len(p), dtype=torch.bool)
    kind, thr = name.split("<") if "<" in name else name.split(">")
    if kind == "nn":
        return nn_info(p, v)[0] < float(thr)
    return DT ** 3 * jerk(p, v, m).norm(dim=-1) / 6 > float(thr)


VARIANTS = [
    ("verlet1", "none", 1, "full", None),
    ("kdk2_uniform", "all", 2, "full", None),
    ("kdk4_uniform(=truth)", "all", 4, "full", None),
    ("adapt_nn<0.5_M4", "nn<0.5", 4, "full", None),
    ("adapt_nn<1_M4", "nn<1", 4, "full", None),
    ("adapt_nn<2_M4", "nn<2", 4, "full", None),
    ("adapt_nn<1_M2", "nn<1", 2, "full", None),
    ("adapt_jerr>1e-4_M4", "jerr>1e-4", 4, "full", None),
    ("adapt_jerr>3e-4_M4", "jerr>3e-4", 4, "full", None),
    ("adapt_jerr>1e-3_M4", "jerr>1e-3", 4, "full", None),
    ("near1_frozen_nn<1_M4", "nn<1", 4, "frozen", 1.0),
    ("near2_frozen_nn<1_M4", "nn<1", 4, "frozen", 2.0),
    ("near4_frozen_nn<1_M4", "nn<1", 4, "frozen", 4.0),
    ("near2_extrap_nn<1_M4", "nn<1", 4, "extrap", 2.0),
    ("near2_frozen_nn<2_M4", "nn<2", 4, "frozen", 2.0),
    ("near2_frozen_all_M4", "all", 4, "frozen", 2.0),
    ("near2_frozen_jerr>3e-4_M4", "jerr>3e-4", 4, "frozen", 2.0),
]


def part_b(scenes, variants):
    res = {}
    for name, crit, M, mode, rc in variants:
        errs, fr, dE, dP = [], [], [], []
        for sc in scenes:
            m = sc["m"]
            p, v = sc["P"][0], sc["V"][0]
            a = acc(p, m)
            far_prev = None
            E0, P0 = energy(p, v, m), (m[:, None] * v).sum(0)
            P = [p]
            for _ in range(TR):
                act = criterion(crit, p, v, m)
                fr.append(float(act.double().mean()))
                p, v, a, far_prev = adaptive_tick(p, v, a, m, act, M, mode, rc, far_prev)
                P.append(p)
            errs.append((torch.stack(P) - sc["P"]).norm(dim=-1).mean(1).numpy())
            dE.append(abs(energy(p, v, m) - E0) / abs(E0))
            dP.append(float(((m[:, None] * v).sum(0) - P0).norm() / (m * v.norm(dim=-1)).sum()))
        cm = np.mean(errs, 0)
        f = float(np.mean(fr))
        res[name] = dict(err={str(k): float(cm[k]) for k in KS}, f=f, M=M, mode=mode, rc=rc,
                         full_evals_per_tick=1 + (M - 1) * (f if mode == "full" else 0.0),
                         promoted_body_substeps_per_tick=(M - 1) * f, dE100_median=float(np.median(dE)),
                         dP100_median=float(np.median(dP)))
    return res


def fast_acc(p, m, act, rc):
    """Pairwise forces of pairs with at least one promoted body, weighted by the smooth split window (1 - (r/rc)^2)^2 (all
    pairs, weight 1, when rc is None). Returns (acc, fast pairs per body)."""
    d, r2, w = pairs(p, m)
    sel = (act[:, None] | act[None, :]).double()
    if rc is not None:
        sel = sel * (1 - (r2 / rc ** 2).clamp(max=1.0)) ** 2
    w = w * sel
    return (d * w[..., None]).sum(1), float((w > 0).double().sum() / len(p))


def respa_tick(p, v, a0, m, act, M, rc):
    """Symmetric multi-rate KDK: slow = full - fast, half-kicked at tick start and end; fast pairs (>= one promoted body,
    window rc) kicked on M substeps for both bodies; all bodies drift every substep. One full eval per tick."""
    h = DT / M
    f, npair = fast_acc(p, m, act, rc)
    v = v + 0.5 * DT * (a0 - f)
    for _ in range(M):
        v = v + 0.5 * h * f
        p = p + h * v
        f, _ = fast_acc(p, m, act, rc)
        v = v + 0.5 * h * f
    an = acc(p, m)
    v = v + 0.5 * DT * (an - f)
    return p, v, an, npair


YO1 = 1 / (2 - 2 ** (1 / 3))
YO0 = -(2 ** (1 / 3)) * YO1


def composed_tick(p, v, a, m, coefs):
    for c in coefs:
        v = v + 0.5 * c * DT * a
        p = p + c * DT * v
        a = acc(p, m)
        v = v + 0.5 * c * DT * a
    return p, v, a


VARIANTS2 = [
    ("verlet1", "none", 1, "respa", 2.0),
    ("respa_all_rcinf_M2(=kdk2)", "all", 2, "respa", None),
    ("kdk3_uniform", None, 3, "comp", None),
    ("yoshida4_uniform", None, 3, "yoshida", None),
    ("jerkdrift_backdiff", None, 1, "jerk_bd", None),
    ("jerkdrift_exactjerk", None, 1, "jerk_ex", None),
    ("respa_all_rc1_M2", "all", 2, "respa", 1.0),
    ("respa_all_rc2_M2", "all", 2, "respa", 2.0),
    ("respa_all_rc4_M2", "all", 2, "respa", 4.0),
    ("respa_all_rc8_M2", "all", 2, "respa", 8.0),
    ("respa_all_rc2_M4", "all", 2 * 2, "respa", 2.0),
    ("respa_all_rc4_M4", "all", 4, "respa", 4.0),
    ("respa_all_rc8_M4", "all", 4, "respa", 8.0),
    ("respa_nn<0.5_rc4_M4", "nn<0.5", 4, "respa", 4.0),
    ("respa_nn<1_rc2_M4", "nn<1", 4, "respa", 2.0),
    ("respa_nn<1_rc4_M4", "nn<1", 4, "respa", 4.0),
    ("respa_nn<1_rc4_M2", "nn<1", 2, "respa", 4.0),
    ("respa_nn<2_rc4_M4", "nn<2", 4, "respa", 4.0),
    ("respa_nn<1_rcinf_M4", "nn<1", 4, "respa", None),
    ("respa_jerr>3e-4_rc4_M4", "jerr>3e-4", 4, "respa", 4.0),
    ("respa_jerr>1e-4_rc4_M4", "jerr>1e-4", 4, "respa", 4.0),
]


def part_b2(scenes, variants):
    res = {}
    for name, crit, M, mode, rc in variants:
        errs, fr, npr, dE, dP = [], [], [], [], []
        for sc in scenes:
            m = sc["m"]
            p, v = sc["P"][0], sc["V"][0]
            a = acc(p, m)
            a_old = None
            E0, P0 = energy(p, v, m), (m[:, None] * v).sum(0)
            P = [p]
            for _ in range(TR):
                if mode == "respa":
                    act = criterion(crit, p, v, m)
                    fr.append(float(act.double().mean()))
                    p, v, a, n = respa_tick(p, v, a, m, act, M, rc)
                    npr.append(n)
                elif mode == "comp":
                    p, v, a = composed_tick(p, v, a, m, [1 / M] * M)
                elif mode == "yoshida":
                    p, v, a = composed_tick(p, v, a, m, [YO1, YO0, YO1])
                else:
                    j = (a - a_old) / DT if (mode == "jerk_bd" and a_old is not None) else (jerk(p, v, m) if mode == "jerk_ex" else 0 * a)
                    pn = p + DT * v + 0.5 * DT ** 2 * a + DT ** 3 * j / 6
                    an = acc(pn, m)
                    v = v + 0.5 * DT * (a + an)
                    p, a_old, a = pn, a, an
                P.append(p)
            errs.append((torch.stack(P) - sc["P"]).norm(dim=-1).mean(1).numpy())
            dE.append(abs(energy(p, v, m) - E0) / abs(E0))
            dP.append(float(((m[:, None] * v).sum(0) - P0).norm() / (m * v.norm(dim=-1)).sum()))
        cm = np.mean(errs, 0)
        full = {"respa": 1, "comp": M, "yoshida": 3}.get(mode, 1)
        res[name] = dict(err={str(k): float(cm[k]) for k in KS}, f=float(np.mean(fr)) if fr else None, M=M, mode=mode, rc=rc,
                         full_evals_per_tick=full, fast_pairs_per_body=float(np.mean(npr)) if npr else None,
                         fast_pair_evals_per_body_tick=float(np.mean(npr)) * (M + 1) if npr else None,
                         dE100_median=float(np.median(dE)), dP100_median=float(np.median(dP)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--b2", action="store_true", help="only part B2 (symmetric multi-rate, higher-order, jerk drift)")
    args = ap.parse_args()
    if args.b2:
        out = {"truth": "fp64 KDK 4 substeps", "dt": DT, "eps": EPS, "ticks": TR}
        for key, (fam, num, seed) in {"expB": ("expB", 24, 9300), "cluster": ("cluster", 12, 9200)}.items():
            sc = make_scenes(fam, 2 if args.smoke else num, seed)
            out[key] = {"partB2": part_b2(sc, VARIANTS2[:6] if args.smoke else VARIANTS2)}
            print(key, "partB2", flush=True)
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=1)
        print("DONE", flush=True)
        return
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()
    fams = {"expB": ("expB", 24, 9300), "cluster": ("cluster", 12, 9200)}
    out = {"truth": "fp64 KDK 4 substeps", "dt": DT, "eps": EPS, "ticks": TR}
    for key, (fam, num, seed) in fams.items():
        sc = make_scenes(fam, 2 if args.smoke else num, seed)
        out[key] = {"N": [int(len(s["m"])) for s in sc], "partA": part_a(sc)}
        print(key, "partA", round(time.time() - t0), flush=True)
        out[key]["partB"] = part_b(sc, VARIANTS[:4] if args.smoke else VARIANTS)
        print(key, "partB", round(time.time() - t0), flush=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
    print("DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
