"""Diagnostics for why the scatter field failed on LJ MD (docs/debugging/scatter-lj.md, section "Diagnosis 2026-10-23").
Usage: PYTHONPATH=. python -m scripts.scatter_lj_diag --ckpt ckpt/ft_final.pt --tag ft
Checks: force at truth states (total / pair term / mesh term vs true LJ, binned by nearest-neighbour distance, velocity dependence, power a.v),
NVE rollout ablations (full, pair-only, mesh-only, velocity channels zeroed, recurrent field zeroed, knn 64), true-force Verlet at the model dt,
tiled vs single-tile forces, knn-24 truncation counts, training-data coverage of the eval states."""
import argparse
import json
import math
import os

import numpy as np
import torch

from scripts import lj_sim
from scripts import scatter_lj as sl


class OptTiles(sl.PeriodicTiles):
    def __init__(self, model, knn=sl.KNN, use_pair=True, use_mesh=True, vzero=False, fz=False):
        super().__init__(model, knn)
        self.use_pair, self.use_mesh, self.vzero, self.fz = use_pair, use_mesh, vzero, fz

    def force(self, pos, vel, mask, box, lay, field, parts=False):
        m = self.m
        S, N, _ = pos.shape
        pk = self.pack(pos, vel, mask, box, lay)
        P, V, K = pk["P"].to(field.dtype), pk["V"].to(field.dtype), pk["K"].to(field.dtype)
        x = m.scatter(P, V * (0.0 if self.vzero else 1.0), K, K)
        ak = m.kernel_acc(x[:, :1])
        f = m.net(torch.cat([x, ak, torch.zeros_like(field) if self.fz else field], 1))
        acc = m.neg_grad(f[:, :1]) + ak
        a_mesh = m.gather(acc, P) * K[..., None]
        a_pp = m.pp_acc(P, K, K, knn=self.knn) * K[..., None]
        um = lambda t: self.unpack(t, pk, S, N).to(pos.dtype)
        a = um(a_mesh * float(self.use_mesh) + a_pp * float(self.use_pair))
        mk = mask[..., None]
        a = (a - (mk * a).sum(1, keepdim=True) / mk.sum(1, keepdim=True)) * mk
        if parts:
            return a, um(a_pp), um(a_mesh)
        return a, f


def demean(a):
    return a - a.mean(1, keepdim=True)


def min_image(p, box):
    d = p[:, None, :] - p[None, :, :]
    return d - box * torch.round(d / box)


def load_scene(name, dev):
    z = np.load(f"results/lj_{name}.npz")
    return z, torch.tensor(z["box"], device=dev, dtype=torch.float64)


def check_forces(model, dev, scenes, frames, warm=8):
    out = {}
    for name in scenes:
        z, box = load_scene(name, dev)
        res = {}
        for fr in frames:
            fr = min(fr, len(z["pos"]) - 1)
            p = torch.tensor(z["pos"][fr], device=dev, dtype=torch.float64)
            v = torch.tensor(z["vel"][fr], device=dev, dtype=torch.float64)
            n = len(p)
            lay = sl.layout((box / sl.LS).cpu().numpy())
            mask = torch.ones(1, n, device=dev, dtype=torch.float64)
            bm = (box / sl.LS)[None]
            pt = OptTiles(model)
            pm, vm = (p / sl.LS)[None], (v / sl.LS)[None]
            st = (pt.init_field(1, lay, dev), None)
            with torch.no_grad():
                for _ in range(warm):
                    pm, vm, st = pt.step(pm, vm, mask, bm, lay, st)
            fld = st[0]
            p, v = pm[0] * sl.LS, vm[0] * sl.LS
            at, _, _ = lj_sim.force(p, box)
            at = demean(at[None])[0]
            with torch.no_grad():
                a, ap, am = pt.force(pm, vm, mask, bm, lay, fld, parts=True)
                a0, _ = pt.force(pm, vm * 0, mask, bm, lay, fld)
                an, _ = pt.force(pm, -vm, mask, bm, lay, fld)
            S = lambda t: demean(t)[0] * sl.LS
            a, ap, am, a0, an = S(a), S(ap), S(am), S(a0), S(an)
            nrm = lambda t: float(t.norm())
            nt = max(nrm(at), 1e-12)
            r = min_image(p, box).norm(dim=-1)
            r.fill_diagonal_(1e9)
            nn = r.min(1).values
            bins = [0, 0.9, 1.0, 1.2, 1.6, 99]
            bd = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = (nn >= lo) & (nn < hi)
                if int(s.sum()) == 0:
                    continue
                bd.append({"nn": [lo, hi], "frac": float(s.float().mean()), "|a_true|": float(at[s].norm(dim=-1).median()),
                           "err_full": float((a - at)[s].norm(dim=-1).median()), "err_pair": float((ap - at)[s].norm(dim=-1).median()),
                           "|mesh|": float(am[s].norm(dim=-1).median())})
            pp_cnt = ((r * 1.0 < 2.0 * sl.LS).sum(1)).float()
            res[fr] = {
                "rel_err_full": nrm(a - at) / nt, "rel_err_pair": nrm(ap - at) / nt, "mesh_over_true": nrm(am) / nt,
                "rel_err_pair_plus_mesh_removed": nrm(ap - at) / nt,
                "v_dep_zero": nrm(a - a0) / max(nrm(a), 1e-12), "v_dep_flip": nrm(a - an) / max(nrm(a), 1e-12), "v_dep_zero_over_true": nrm(a - a0) / nt,
                "power_true": float((at * v).sum()) / n, "power_full": float((a * v).sum()) / n, "power_pair": float((ap * v).sum()) / n,
                "power_mesh": float((am * v).sum()) / n, "ke_per_n": float(0.5 * (v ** 2).sum()) / n,
                "drag_gamma": -float(((a - at) * v).sum() / (v ** 2).sum()),
                "mean_neighbors_in_rc": float(pp_cnt.mean()), "frac_over_knn24": float((pp_cnt > sl.KNN).float().mean()),
                "frac_nn_below_0.9": float((nn < 0.9).float().mean()), "bins": bd}
        out[name] = res
        print(name, {k: round(v, 4) for k, v in res[min(frames[-1], len(z["pos"]) - 1)].items() if not isinstance(v, list)}, flush=True)
    return out


@torch.no_grad()
def roll(model, z, dev, frames, **opts):
    box = torch.tensor(z["box"], device=dev, dtype=torch.float64)
    pt = OptTiles(model, **opts)
    n = z["pos"].shape[1]
    lay = sl.layout((box / sl.LS).cpu().numpy())
    p = torch.tensor(z["pos"][0] / sl.LS, device=dev, dtype=torch.float64)[None]
    v = torch.tensor(z["vel"][0] / sl.LS, device=dev, dtype=torch.float64)[None]
    bm = (box / sl.LS)[None]
    mask = torch.ones(1, n, device=dev, dtype=torch.float64)
    st = (pt.init_field(1, lay, dev), None)
    area = float(box[0] * box[1])
    series = []
    for j in range(frames * 4 + 1):
        if j % 40 == 0:
            s = sl.true_stats(p[0] * sl.LS, v[0] * sl.LS, box, area)
            series.append([j // 4] + s)
            if not math.isfinite(s[2]):
                break
        if j == frames * 4:
            break
        p, v, st = pt.step(p, v, mask, bm, lay, st)
    return series


def check_rollouts(model, dev, scenes, frames):
    variants = {"full": {}, "pair_only": {"use_mesh": False}, "mesh_only": {"use_pair": False}, "vzero": {"vzero": True},
                "field_zero": {"fz": True}, "knn64": {"knn": 64}}
    out = {}
    for name in scenes:
        z = np.load(f"results/lj_{name}.npz")
        o = {"truth_E": z["stats"][:frames + 1:10, 3].tolist(), "truth_T": z["stats"][:frames + 1:10, 1].tolist()}
        for vn, kw in variants.items():
            ser = roll(model, z, dev, frames, **kw)
            o[vn] = {"T": [s[1] for s in ser], "E": [s[3] for s in ser], "P": [s[4] for s in ser]}
            print(name, vn, "E", np.round(o[vn]["E"], 3).tolist(), "T", np.round(o[vn]["T"], 3).tolist(), flush=True)
        out[name] = o
    return out


def check_integrator(dev, scenes, frames):
    out = {}
    for name in scenes:
        z = np.load(f"results/lj_{name}.npz")
        box = z["box"]
        res = {}
        for dt in (0.005, 0.025):
            rec = int(round(0.1 / dt)) * 10
            _, _, st = lj_sim.run(z["pos"][0].astype(np.float64), z["vel"][0].astype(np.float64), box, int(round(frames * 0.1 / dt)), rec, dt, dev)
            res[str(dt)] = {"E": st[:, 3].tolist(), "T": st[:, 1].tolist()}
        out[name] = res
        print("integrator", name, {k: np.round(v["E"], 4).tolist() for k, v in res.items()}, flush=True)
    return out


@torch.no_grad()
def check_tiling(model, dev):
    rng = np.random.default_rng(4738)
    pos, box = lj_sim.lattice(22, 22, 0.7, 0.1, rng)
    vel = lj_sim.maxwell(len(pos), 0.6, rng)
    p = torch.tensor(pos / sl.LS, device=dev, dtype=torch.float64)[None]
    v = torch.tensor(vel / sl.LS, device=dev, dtype=torch.float64)[None]
    bm = torch.tensor(box / sl.LS, device=dev, dtype=torch.float64)[None]
    mask = torch.ones(1, len(pos), device=dev, dtype=torch.float64)
    pt = OptTiles(model)
    res = {"box_model": bm[0].tolist(), "n": len(pos)}
    outs = {}
    for lay in ((1, 1), (2, 2), (3, 3)):
        a, _ = pt.force(p, v, mask, bm, lay, pt.init_field(1, lay, dev))
        outs[lay] = a
    for lay in ((2, 2), (3, 3)):
        d = (outs[lay] - outs[(1, 1)]).norm(dim=-1)
        res[f"tiles{lay}_vs_1x1"] = {"max_abs": float(d.max()), "rel_l2": float((outs[lay] - outs[(1, 1)]).norm() / outs[(1, 1)].norm())}
    print("tiling", res, flush=True)
    return res


@torch.no_grad()
def check_twobody_v(model, dev):
    pt = OptTiles(model)
    out = []
    for dd in (0.9, 1.0, 1.12, 1.4, 1.8, 2.2):
        d = dd / sl.LS
        pos = torch.tensor([[[15 - d / 2, 15.0], [15 + d / 2, 15.0]]], device=dev, dtype=torch.float64)
        box = torch.full((1, 2), 30.0, device=dev, dtype=torch.float64)
        mask = torch.ones(1, 2, device=dev, dtype=torch.float64)
        lay = sl.layout(box[0])
        row = {"d": dd}
        for tag, vv in (("v0", [[0, 0], [0, 0]]), ("par", [[1, 0], [-1, 0]]), ("perp", [[0, 1], [0, -1]]), ("com", [[1, 0.5], [1, 0.5]])):
            vel = torch.tensor([vv], device=dev, dtype=torch.float64)
            a, _ = pt.force(pos, vel, mask, box, lay, pt.init_field(1, lay, dev))
            row[tag] = (a[0, 0] * sl.LS).tolist()
        out.append(row)
    print("twobody_v", out, flush=True)
    return out


def check_coverage(cache, dev):
    if not os.path.exists(cache):
        return None
    z = np.load(cache)
    P, V, n, box = z["tr_P"], z["tr_V"], z["tr_n"], z["tr_box"]
    rho = n / (box[:, 0] * box[:, 1])
    T = np.array([0.5 * (V[i, :, : n[i]] ** 2).sum((1, 2)).mean() / n[i] for i in range(len(n))])
    out = {"n_scenes": int(len(n)), "rho_quantiles": np.quantile(rho, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
           "T_quantiles": np.quantile(T, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()}
    for name, (r0, t0) in {"gas": (0.05, 1.12), "liquid": (0.7, 0.5), "solid": (0.95, 0.214), "coexist": (0.35, 0.98), "melt_start": (0.9, 0.1),
                           "melt_end": (0.9, 1.05), "quench_start": (0.8, 1.2), "quench_end": (0.8, 0.29)}.items():
        near = (np.abs(np.log(rho / r0)) < 0.15) & (np.abs(np.log(T / t0)) < 0.3)
        out[f"cover_{name}"] = int(near.sum())
    out["T_below_0.1"] = int((T < 0.1).sum())
    print("coverage", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--cache", default="ckpt/data_s4738_n320_f100_t2.5_h0.15.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = sl.build_model(None, dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    skip = args.skip.split(",")
    out = {}
    if "forces" not in skip:
        out["forces"] = check_forces(model, dev, list(lj_sim.SCENES), [0, 100, 400, 800])
    if "twobody" not in skip:
        out["twobody_v"] = check_twobody_v(model, dev)
    if "tiling" not in skip:
        out["tiling"] = check_tiling(model, dev)
    if "coverage" not in skip:
        out["coverage"] = check_coverage(args.cache, dev)
    if "integrator" not in skip:
        out["integrator"] = check_integrator(dev, ["gas", "liquid", "solid", "coexist"], args.frames)
    if "rollouts" not in skip:
        out["rollouts"] = check_rollouts(model, dev, ["gas", "liquid", "solid", "coexist"], args.frames)
    json.dump(out, open(f"results/scatter_lj_diag_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
