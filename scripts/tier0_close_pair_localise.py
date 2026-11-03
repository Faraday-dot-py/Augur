"""Localise the tiled closed-loop tier 0's potential-energy loss (24 scenes x 100 ticks per seed 9100/9200/9300, same scenes as
scripts/tier0_energy_metric_audit.py). Per method (baseline_tiled, matched60k_halo, new_halo, oracle exact force through the dt 0.1 Verlet step):
  pe_bins: potential energy of pairs binned by raw separation r (edges 0.5, 1, 2, 4), model minus truth at the same tick, / |E_truth(0)|, plus pair-count ratio
  density_bins: per-particle potential energy binned by neighbour count within r<4 (truth-state terciles), model minus truth
  bound_pairs: pairs with truth r<1 at tick 20: fraction still r<1 at later ticks in truth and in the model, and mean log(r_model / r_truth)
  teacher_force: truth states fed to the force every tick (recurrent field carried): projected force gain <a_model, a_true>/|a_true|^2 and relative error,
    binned by truth nearest-neighbour distance
Usage: PYTHONPATH=. python scripts/tier0_close_pair_localise.py --m60k <ckpt> --new <ckpt> --out <json>"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import tier0_tiled_closed_loop as tc
from scripts import tiered_adapters as ta
from scripts import train_tier0_ext as tte
from scripts import train_tiered as ttr
from scripts.tier0_energy_metric_audit import roll_states

SEEDS = (9100, 9200, 9300)
DT, EPS = ttr.DT, ttr.EPS
EDGES = [0.0, 0.5, 1.0, 2.0, 4.0, 1e9]
TICKS = (10, 20, 40, 60, 100)


def pair_geom(pos, mask):
    pos = pos.double()
    n = pos.shape[1]
    d = pos[:, None] - pos[:, :, None]
    r2 = (d ** 2).sum(-1)
    ok = (mask[:, :, None] * mask[:, None, :]).double() * (1 - torch.eye(n, device=pos.device, dtype=torch.float64))
    return r2.sqrt(), ok


def pe_bins(pos, mask):
    r, ok = pair_geom(pos, mask)
    w = 0.5 / torch.sqrt(r ** 2 + EPS ** 2) * ok
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = ((r >= lo) & (r < hi)).double()
        out.append((-(w * sel).sum((-1, -2)), 0.5 * (ok * sel).sum((-1, -2))))
    return out


def per_particle_pe(pos, mask):
    r, ok = pair_geom(pos, mask)
    return -0.5 * (ok / torch.sqrt(r ** 2 + EPS ** 2)).sum(-1), ((r < 4.0).double() * ok).sum(-1)


def summarise(P, V, truth_P, truth_V, mask, E0abs, ticks=TICKS):
    out = {"pe_bins": {}, "density_bins": {}, "bound_pairs": {}}
    rt20, ok = pair_geom(truth_P[:, 20], mask)
    bound = ((rt20 < 1.0) & (ok > 0)).double()
    for k in ticks:
        mb, tb = pe_bins(P[:, k], mask), pe_bins(truth_P[:, k], mask)
        out["pe_bins"][str(k)] = [{"r": f"{lo}-{hi}", "dPE_over_E0": float(((m[0] - t[0]) / E0abs).mean()), "model_PE_over_E0": float((m[0] / E0abs).mean()),
                                   "truth_PE_over_E0": float((t[0] / E0abs).mean()), "pair_count_ratio": float(m[1].sum() / t[1].sum().clamp_min(1))}
                                  for (lo, hi), m, t in zip(zip(EDGES[:-1], EDGES[1:]), mb, tb)]
        pm, nm = per_particle_pe(P[:, k], mask)
        pt, nt = per_particle_pe(truth_P[:, k], mask)
        sel = mask > 0
        qs = torch.quantile(nt[sel], torch.tensor([1 / 3, 2 / 3], dtype=torch.float64, device=nt.device))
        rows = []
        for name, lo, hi in (("low", -1, qs[0]), ("mid", qs[0], qs[1]), ("high", qs[1], 1e9)):
            sm, st = sel & (nm > lo) & (nm <= hi), sel & (nt > lo) & (nt <= hi)
            rows.append({"bin": name, "dPE_per_particle_over_E0": float(((pm * sm).sum() - (pt * st).sum()) / E0abs.sum()),
                         "n_model": int(sm.sum()), "n_truth": int(st.sum())})
        out["density_bins"][str(k)] = rows
        rm, _ = pair_geom(P[:, k], mask)
        rtk, _ = pair_geom(truth_P[:, k], mask)
        nb = bound.sum().clamp_min(1)
        lr = torch.log((rm + 0.05) / (rtk + 0.05))
        out["bound_pairs"][str(k)] = {"n_pairs": float(bound.sum() / 2), "frac_still_bound_truth": float(((rtk < 1.0).double() * bound).sum() / nb),
                                      "frac_still_bound_model": float(((rm < 1.0).double() * bound).sum() / nb), "mean_log_r_model_over_truth": float((lr * bound).sum() / nb)}
    return out


def teacher_forced(force_fn, init_state, d, ticks):
    m, mk = d["M"].float(), d["mask"].float()
    st = init_state(len(m))
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
    acc = {k: {b: [0.0, 0.0, 0.0, 0] for b in bins} for k in ticks}
    for t in range(max(ticks) + 1):
        pos, vel = d["P"][:, t].float(), d["V"][:, t].float()
        a, st = force_fn(pos, vel, m, mk, st)
        if t in ticks:
            at = ttr.true_acc(pos, m, mk)
            r, ok = pair_geom(pos, mk)
            nn = (r + 1e9 * (1 - ok)).amin(-1)
            gain = (a * at).sum(-1) / (at ** 2).sum(-1).clamp_min(1e-12)
            rel = ((a - at) ** 2).sum(-1).sqrt() / at.norm(dim=-1).clamp_min(1e-6)
            for b in bins:
                s = (nn >= b[0]) & (nn < b[1]) & (mk > 0)
                acc[t][b][0] += float(gain[s].sum())
                acc[t][b][1] += float(rel[s].sum())
                acc[t][b][3] += int(s.sum())
    return {str(k): [{"nn_dist": f"{b[0]}-{b[1]}", "n": v[3], "mean_force_gain": v[0] / max(v[3], 1), "mean_rel_force_err": v[1] / max(v[3], 1)} for b, v in acc[k].items()]
            for k in ticks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0", default="E_ms_kp_pot_v_g128.pt")
    ap.add_argument("--m60k", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, path in (("matched60k", a.m60k), ("new", a.new)):
        m = tte.build(argparse.Namespace(t0="ext=6,extapply,taper=8,bnd,halo=8", dt=0.1), dev)
        sf.load_compat(m, torch.load(path, map_location=dev, weights_only=False)["model"])
        models[name] = m.eval().requires_grad_(False)
    bl = ta.load_tiled(f"{a.tier0},nt=4,tile=32,halo=16,far=1", dev)
    res = {}
    for sd in SEEDS:
        d = ttr.make_scenes(a.scenes, a.steps, sd, dev)
        E0abs = pe_bins(d["P"][:, 0], d["mask"])
        E0abs = (0.5 * (d["M"] * d["mask"] * (d["V"][:, 0] ** 2).sum(-1)).sum(-1) - sum(b[0] for b in E0abs)).abs()
        funcs = {"oracle_force_dt0.1_verlet": (lambda p, v, m, k, s: (ttr.true_acc(p, m, k), s), lambda S: None),
                 "baseline_tiled": (lambda p, v, m, k, s: bl.force(p, v, m, k, s), lambda S: bl.init_state(S, dev))}
        for name, key in (("matched60k_halo", "matched60k"), ("new_halo", "new")):
            tt = tc.TiledTier0(models[key], dev, chunk=a.chunk, exchange="halo")
            funcs[name] = (lambda p, v, m, k, s, tt=tt: tt.force(p, v, m, k, s), tt.init_state)
        res[str(sd)] = {"E0abs_mean": float(E0abs.mean())}
        for n, (f, i0) in funcs.items():
            P, V = roll_states(f, i0, d, a.steps)
            res[str(sd)][n] = summarise(P, V, d["P"], d["V"], d["mask"], E0abs)
            if n != "oracle_force_dt0.1_verlet":
                res[str(sd)][n]["teacher_force"] = teacher_forced(f, i0, d, (10, 20, 40))
        nb = res[str(sd)]["new_halo"]["pe_bins"]["100"]
        print(sd, json.dumps([(r["r"], round(r["dPE_over_E0"], 4), round(r["pair_count_ratio"], 3)) for r in nb]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
