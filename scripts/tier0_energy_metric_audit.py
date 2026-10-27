"""Metric audit of dE@100 (scripts/exp12_hier.eval_generic: mean over scenes of |E_model(100) - E_truth(0)| / |E_truth(0)|, E = global pair energy
over all particles, eps 0.5, no windowing). Rolls the same held-out scenes (24 scenes x 100 ticks, seeds 9100/9200/9300) with each force function
and records per tick: total/kinetic/potential energy, potential split into same-tile and cross-tile pairs and into r<4 / r>=4 pairs, the signed
drift, particles outside the 128-unit domain, and the same quantities for the truth trajectory and for an oracle force (exact pairwise acceleration)
through the identical single-step dt 0.1 Verlet. Usage: PYTHONPATH=. python scripts/tier0_energy_metric_audit.py --out <json>"""
import argparse
import json

import numpy as np
import torch

from scripts import exp12_hier as h
from scripts import scatter_field as sf
from scripts import tier0_tiled_closed_loop as tc
from scripts import tiered_adapters as ta
from scripts import train_tier0_ext as tte
from scripts import train_tiered as ttr

SEEDS = (9100, 9200, 9300)
DT, EPS = ttr.DT, ttr.EPS


def parts(pos, vel, mass, mask):
    pos, vel = pos.double(), vel.double()
    mm = (mass * mask).double()
    n = pos.shape[1]
    d = pos[:, None] - pos[:, :, None]
    r = torch.sqrt((d ** 2).sum(-1) + EPS ** 2)
    w = mm[:, :, None] * mm[:, None, :] / r * (1 - torch.eye(n, device=pos.device, dtype=pos.dtype))
    own = h.owner(pos)
    same = (own[:, :, None] == own[:, None, :]).double()
    near = (r < 4.0).double()
    ke = 0.5 * (mm * (vel ** 2).sum(-1)).sum(-1)
    pe = -0.5 * w.sum((-1, -2))
    out = {"KE": ke, "PE": pe, "E": ke + pe, "PE_same_tile": -0.5 * (w * same).sum((-1, -2)), "PE_cross_tile": -0.5 * (w * (1 - same)).sum((-1, -2)),
           "PE_r_lt4": -0.5 * (w * near).sum((-1, -2)), "PE_r_ge4": -0.5 * (w * (1 - near)).sum((-1, -2))}
    out["outside"] = (((pos.abs().amax(-1) > 64.0).double()) * mask.double()).sum(-1)
    out["speed_max"] = (vel.norm(dim=-1) * mask.double()).amax(-1)
    return out


@torch.no_grad()
def roll(force_fn, init_state, d, T):
    m, mk = d["M"].float(), d["mask"].float()
    pos, vel = d["P"][:, 0].float(), d["V"][:, 0].float()
    st = init_state(len(pos))
    a_prev, st = force_fn(pos, vel, m, mk, st)
    rec = [parts(pos, vel, d["M"], d["mask"])]
    for t in range(1, T + 1):
        pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
        a_new, st = force_fn(pn, vel + DT * a_prev, m, mk, st)
        pos, vel, a_prev = pn, vel + 0.5 * DT * (a_prev + a_new), a_new
        rec.append(parts(pos, vel, d["M"], d["mask"]))
    return {k: torch.stack([r[k] for r in rec], 1) for k in rec[0]}


def summarise(rec, truth, ticks=(10, 20, 50, 100)):
    E0 = truth["E"][:, 0]
    out = {}
    for k in ticks:
        dE = rec["E"][:, k] - E0
        sim = rec["E"][:, k] - truth["E"][:, k]
        out[str(k)] = {"dE_vs_E0_mean": float((dE.abs() / E0.abs()).mean()), "dE_vs_E0_median": float((dE.abs() / E0.abs()).median()),
                       "signed_vs_E0_mean": float((dE / E0.abs()).mean()), "frac_scenes_gain": float((dE > 0).double().mean()),
                       "dE_vs_Etruth_t_mean": float((sim.abs() / E0.abs()).mean()), "signed_vs_Etruth_t_mean": float((sim / E0.abs()).mean()),
                       "dKE_vs_truth_t": float(((rec["KE"][:, k] - truth["KE"][:, k]) / E0.abs()).mean()),
                       "dPE_vs_truth_t": float(((rec["PE"][:, k] - truth["PE"][:, k]) / E0.abs()).mean()),
                       "dPE_same_tile_vs_truth_t": float(((rec["PE_same_tile"][:, k] - truth["PE_same_tile"][:, k]) / E0.abs()).mean()),
                       "dPE_cross_tile_vs_truth_t": float(((rec["PE_cross_tile"][:, k] - truth["PE_cross_tile"][:, k]) / E0.abs()).mean()),
                       "dPE_r_lt4_vs_truth_t": float(((rec["PE_r_lt4"][:, k] - truth["PE_r_lt4"][:, k]) / E0.abs()).mean()),
                       "dPE_r_ge4_vs_truth_t": float(((rec["PE_r_ge4"][:, k] - truth["PE_r_ge4"][:, k]) / E0.abs()).mean()),
                       "outside_mean": float(rec["outside"][:, k].mean()), "speed_max_mean": float(rec["speed_max"][:, k].mean())}
    out["profile_signed_vs_E0"] = (((rec["E"] - E0[:, None]) / E0.abs()[:, None]).mean(0)).tolist()[::5]
    out["profile_abs_vs_E0"] = (((rec["E"] - E0[:, None]).abs() / E0.abs()[:, None]).mean(0)).tolist()[::5]
    out["profile_signed_vs_Etruth_t"] = (((rec["E"] - truth["E"]) / E0.abs()[:, None]).mean(0)).tolist()[::5]
    return out


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
    t0s = "ext=6,extapply,taper=8,bnd,halo=8"
    models = {}
    for name, path in (("matched60k", a.m60k), ("new", a.new)):
        m = tte.build(argparse.Namespace(t0=t0s, dt=0.1), dev)
        st = torch.load(path, map_location=dev, weights_only=False)
        sf.load_compat(m, st["model"])
        models[name] = m.eval().requires_grad_(False)
    bl = ta.load_tiled(f"{a.tier0},nt=4,tile=32,halo=16,far=1", dev)
    res = {}
    for sd in SEEDS:
        d = ttr.make_scenes(a.scenes, a.steps, sd, dev)
        truth = {k: torch.stack([parts(d["P"][:, t], d["V"][:, t], d["M"], d["mask"])[k] for t in range(a.steps + 1)], 1)
                 for k in parts(d["P"][:, 0], d["V"][:, 0], d["M"], d["mask"])}
        runs = {"oracle_force_dt0.1_verlet": roll(lambda p, v, m, k, s: (ttr.true_acc(p, m, k), s), lambda S: None, d, a.steps),
                "baseline_tiled": roll(lambda p, v, m, k, s: bl.force(p, v, m, k, s), lambda S: bl.init_state(S, dev), d, a.steps)}
        for name, ex in (("matched60k_noexchange", "none"), ("matched60k_halo", "halo"), ("new_halo", "halo")):
            mdl = models["new" if name.startswith("new") else "matched60k"]
            tt = tc.TiledTier0(mdl, dev, chunk=a.chunk, exchange=ex)
            runs[name] = roll(lambda p, v, m, k, s, tt=tt: tt.force(p, v, m, k, s), tt.init_state, d, a.steps)
        res[str(sd)] = {"truth_self": summarise(truth, truth)}
        res[str(sd)]["truth_E0"] = float(truth["E"][:, 0].mean())
        for n, r in runs.items():
            res[str(sd)][n] = summarise(r, truth)
        b = res[str(sd)]["baseline_tiled"]["100"]["dE_vs_E0_mean"]
        print(sd, json.dumps({n: {"dE100": round(v["100"]["dE_vs_E0_mean"], 5), "x_baseline": round(v["100"]["dE_vs_E0_mean"] / b, 2),
                                  "signed": round(v["100"]["signed_vs_E0_mean"], 5), "sim": round(v["100"]["dE_vs_Etruth_t_mean"], 5)} for n, v in res[str(sd)].items()
                                 if isinstance(v, dict)}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
