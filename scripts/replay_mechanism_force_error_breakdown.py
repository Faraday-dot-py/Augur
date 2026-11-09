"""Where the force error of a tiled closed-loop tier 0 lives, on states the model visits (and on exact-trajectory states, and on another checkpoint's visited
states), broken down by local density, radial vs tangential component, mesh vs pair term, tick, distance to the nearest tile edge, and recurrent-field
dependence; plus energy bias vs variance against the exact trajectory's E(t) (never E(0)).
Usage: PYTHONPATH=. python scripts/replay_mechanism_force_error_breakdown.py --ckpt before=<pt> after=<pt> [more name=path] --out <json>
Decomposition identities (all sums over bodies at the recorded ticks): |err|^2 = |mesh - true|^2 + 2<pair, mesh - true> + |pair|^2 where pair is the captured
pair-term output (scatter_field.pp_acc, before the tier-0 head) and mesh = total - pair (mesh net + external field + head). err = model - exact direct sum."""
import argparse
import json
import os

import numpy as np
import torch

from scripts import exp12_hier as h
from scripts import scatter_field as sf
from scripts import train_tiered as ttr
from scripts import tier0_tiled_closed_loop as tc
from scripts.eval_tier0_tiled_closed_loop import load

SEEDS = (9100, 9200, 9300)
R_DENS = 2.0
DENS_EDGES = (2, 5, 10, 20)
EDGE_EDGES = (1.0, 2.0, 4.0, 8.0)
DT, EPS = ttr.DT, ttr.EPS


class PairCapture:
    def __init__(self, model):
        self.orig, self.buf = model.pp_acc, []
        model.pp_acc = self.call

    def call(self, *a, **k):
        out = self.orig(*a, **k)
        self.buf.append(out)
        return out

    def take(self):
        assert self.buf, "pair term was not called: model has no pair term or pp_acc is not on the force path"
        out = torch.cat(self.buf)
        self.buf = []
        return out


def edge_dist(pos):
    x = (pos + h.NT * h.TILE / 2) % h.TILE
    return torch.minimum(x, h.TILE - x).amin(-1)


def density(pos, mk):
    d = torch.cdist(pos, pos)
    near = (d < R_DENS) & (mk[:, None, :] > 0)
    return near.sum(-1) - 1


def radial_unit(pos, m, mk, lab):
    r = torch.zeros_like(pos)
    for c in range(4):
        mc = mk * (lab == c).float()
        com = (pos * (m * mc)[..., None]).sum(1) / (m * mc).sum(1).clamp_min(1e-9)[:, None]
        d = pos - com[:, None]
        r = r + d * mc[..., None]
    n = r.norm(dim=-1, keepdim=True)
    return r / n.clamp_min(1e-9)


def pair_of(tt, cap, pos, S, N):
    p = cap.take().reshape(S, h.NW, N, 2)
    return torch.gather(p, 1, h.owner(pos)[:, None, :, None].expand(S, 1, N, 2)).squeeze(1)


def row_block(t, a, at, pr, anf, pn, m, mk, lab):
    S, N = mk.shape
    ok = (mk > 0) & (lab >= 0)
    rh = radial_unit(pn, m, mk, lab)
    ah = at / at.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    e = a - at
    mesh = a - pr
    f = lambda x: x[ok].double().cpu().numpy()
    g = lambda x: x[ok].double().cpu().numpy()
    out = {"tick": np.full(int(ok.sum()), t, dtype=np.int32), "dens": g(density(pn, mk)), "edge": g(edge_dist(pn)),
           "e": f(e), "at": f(at), "pair": f(pr), "mesh_err": f(mesh - at), "nofld": f(a - anf), "rh": f(rh), "ah": f(ah)}
    return out


@torch.no_grad()
def collect(tt, cap, d, mode, ticks, ext=None):
    m, mk = d["M"].float(), d["mask"].float()
    P, V, lab = d["P"].float(), d["V"].float(), d["lab"]
    S, T = P.shape[:2]
    N = mk.shape[1]
    zero = tt.init_state(S)
    fld = tt.init_state(S)
    a_prev, fld = tt.force(P[:, 0], V[:, 0], m, mk, fld)
    cap.take()
    pos, vel = P[:, 0], V[:, 0]
    at_prev = ttr.true_acc(P[:, 0], m, mk)
    rows, states, en = [], {}, []
    Pd, Vd = d["P"], d["V"]
    en.append(sf.energy(pos.double(), vel.double(), d["M"], d["mask"], EPS).cpu())
    for t in range(1, T):
        if mode == "own":
            pn = pos + DT * vel + 0.5 * DT ** 2 * a_prev
            vin = vel + DT * a_prev
            states[t] = (pn, vin)
        elif mode == "truth":
            pn = P[:, t]
            vin = V[:, t - 1] + DT * at_prev
        else:
            pn, vin = ext[t]
        a_new, fld_new = tt.force(pn, vin, m, mk, fld)
        pr = pair_of(tt, cap, pn, S, N)
        if t in ticks:
            at = ttr.true_acc(pn, m, mk)
            anf, _ = tt.force(pn, vin, m, mk, zero)
            cap.take()
            rows.append(row_block(t, a_new, at, pr, anf, pn, m, mk, lab))
        if mode == "own":
            pos, vel, a_prev = pn, vel + 0.5 * DT * (a_prev + a_new), a_new
            en.append(sf.energy(pos.double(), vel.double(), d["M"], d["mask"], EPS).cpu())
        elif mode == "truth":
            at_prev = ttr.true_acc(pn, m, mk)
        fld = fld_new
    rows = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return rows, states, (torch.stack(en) if en[1:] else None)


def summarise(r, sel):
    e, at, pair, me, rh, ah, nf = (r[k][sel] for k in ("e", "at", "pair", "mesh_err", "rh", "ah", "nofld"))
    sq = lambda x: float((x ** 2).sum())
    ee = (e ** 2).sum(-1)
    er = (e * rh).sum(-1)
    ep = (e * ah).sum(-1)
    return {"n": int(sel.sum()), "sse_err": float(ee.sum()), "sse_true": sq(at), "sse_radial": sq(er), "sse_tangential": float((ee - er ** 2).sum()),
            "sse_parallel_to_true": sq(ep), "sse_perpendicular_to_true": float((ee - ep ** 2).sum()), "bias_along_true": float(ep.sum()),
            "pair_sq": sq(pair), "mesh_err_sq": sq(me), "cross_2_pair_meshErr": float(2 * (pair * me).sum()), "recurrent_field_sensitivity_sq": sq(nf)}


def table(r, ticksel, key, edges):
    out, tot = {}, summarise(r, ticksel)
    b = np.digitize(r[key], edges)
    for i in range(len(edges) + 1):
        lab = f"<{edges[0]}" if i == 0 else f">={edges[-1]}" if i == len(edges) else f"{edges[i - 1]}-{edges[i]}"
        s = summarise(r, ticksel & (b == i))
        s["share_of_group_sse_err"] = s["sse_err"] / max(tot["sse_err"], 1e-30)
        s["share_of_bodies"] = s["n"] / max(tot["n"], 1)
        s["rel_force_err_in_bin"] = (s["sse_err"] / max(s["sse_true"], 1e-30)) ** 0.5 if s["n"] else None
        out[lab] = s
    return out


def full_report(r, split):
    late, early = r["tick"] >= split, r["tick"] < split
    rep = {"overall_rel_force_err": (summarise(r, np.ones_like(late))["sse_err"] / summarise(r, np.ones_like(late))["sse_true"]) ** 0.5}
    for gname, g in (("late_ticks", late), ("early_ticks", early)):
        if not g.any():
            continue
        rep[gname] = {"total": summarise(r, g), "by_density": table(r, g, "dens", DENS_EDGES), "by_tile_edge_distance": table(r, g, "edge", EDGE_EDGES)}
    rep["by_tick"] = {}
    for t in sorted(set(r["tick"].tolist())):
        s = summarise(r, r["tick"] == t)
        rep["by_tick"][str(t)] = {"rel_force_err": (s["sse_err"] / s["sse_true"]) ** 0.5, "bias_along_true_per_body": s["bias_along_true"] / s["n"]}
    return rep


def energy_report(en_list, tt_list, Et_list):
    out = {}
    for t in (20, 60, 100):
        if t >= en_list[0].shape[0]:
            continue
        s = torch.cat([(e[t] - et[t]) / et[t].abs().clamp_min(1e-6) for e, et in zip(en_list, Et_list)])
        s0 = torch.cat([(e[t] - et[0]).abs() / et[0].abs() for e, et in zip(en_list, Et_list)])
        tr = torch.cat([(et[t] - et[0]) / et[0].abs() for et in Et_list])
        out[str(t)] = {"bias_vs_truth_Et": float(s.mean()), "scene_std_vs_truth_Et": float(s.std()), "rms_vs_truth_Et": float((s ** 2).mean().sqrt()),
                       "mean_abs_vs_truth_Et": float(s.abs().mean()), "mean_abs_vs_E0_existing_metric": float(s0.mean()),
                       "truth_drift_Et_over_E0_mean": float(tr.mean()), "truth_drift_abs_mean": float(tr.abs().mean()), "n_scenes": int(s.numel())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path, at least one; every pair is cross-evaluated on the other's visited states")
    ap.add_argument("--exchange", choices=["halo", "zero", "none"], default="halo")
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tick-from", type=int, default=20)
    ap.add_argument("--split", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticks = set(range(a.tick_from, a.steps + 1, 5))
    names, paths = zip(*(c.split("=", 1) for c in a.ckpt))
    for p in paths:
        assert os.path.exists(p), f"MISSING {p}"
    models, caps, tts = {}, {}, {}
    for n, p in zip(names, paths):
        models[n] = load(p, dev)
        caps[n] = PairCapture(models[n])
        tts[n] = tc.TiledTier0(models[n], dev, chunk=a.chunk, exchange=a.exchange, ckpt=False)
    res = {"args": vars(a), "ticks": sorted(ticks), "force_error": {n: {} for n in names}, "energy": {n: {} for n in names}, "per_seed_overall_rel_force_err": {n: {} for n in names}}
    pooled = {(n, s): [] for n in names for s in ["own", "truth"] + [f"states-of-{o}" for o in names if o != n]}
    own_states, own_en, own_truth_en = {}, {n: [] for n in names}, {n: [] for n in names}
    for sd in SEEDS:
        d = ttr.make_scenes(a.scenes, a.steps, sd, dev)
        Et = torch.stack([sf.energy(d["P"][:, t], d["V"][:, t], d["M"], d["mask"], EPS).cpu() for t in range(d["P"].shape[1])])
        for n in names:
            rows, st, en = collect(tts[n], caps[n], d, "own", ticks)
            pooled[(n, "own")].append(rows)
            own_states[n] = st
            own_en[n].append(en)
            own_truth_en[n].append(Et)
            res["per_seed_overall_rel_force_err"][n][str(sd)] = (summarise(rows, np.ones(len(rows["tick"]), bool))["sse_err"] / summarise(rows, np.ones(len(rows["tick"]), bool))["sse_true"]) ** 0.5
            print(f"{n} own {sd} rel force err {res['per_seed_overall_rel_force_err'][n][str(sd)]:.4f}", flush=True)
        for n in names:
            pooled[(n, "truth")].append(collect(tts[n], caps[n], d, "truth", ticks)[0])
            for o in names:
                if o != n:
                    pooled[(n, f"states-of-{o}")].append(collect(tts[n], caps[n], d, "states", ticks, ext=own_states[o])[0])
        del own_states
        own_states = {}
    for (n, s), lst in pooled.items():
        r = {k: np.concatenate([x[k] for x in lst]) for k in lst[0]}
        res["force_error"][n][s] = full_report(r, a.split)
        if s == "own" and not (r["pair"] ** 2).sum() > 0:
            print(f"WARNING {n}: captured pair term is identically zero", flush=True)
    for n in names:
        res["energy"][n] = energy_report(own_en[n], None, own_truth_en[n])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
