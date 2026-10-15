"""State-assimilation probe on frozen Exp E (E_ms_kp_pot_v_g128): inference only, no weight updates. See docs/debugging/few-shot-adaptation.md.

  PYTHONPATH=. python scripts/scatter_assim_probe.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/assim/assim.json
Observation = truth position/velocity of a body set, written into the rollout state at chosen ticks; the model's a_prev is recomputed at the written state.
Families: solar (scatter_solar.solar_scene k=4.0, seed 4738, 1000 ticks), star (mass-ratio OOD, seed 9200, 150 ticks), nr100 (100 unit-mass bodies, seed 9300, 1000 ticks).
Error = mean per-body position error (sim units) averaged over a tick window (sawtooth-fair; a plain err@k lands on reset ticks).
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss
from scripts import scatter_solar as so
from scripts.scatter_error_growth import DEV, DT, EPS, Net, exact_traj

SEED = 4738
WINDOWS = {"20": (11, 20), "100": (81, 100), "300": (281, 300), "1000": (981, 1000)}
# arm: (reset period K or 0, first-m ticks or 0, observed set, pos_only)
ARMS = {
    "free": (0, 0, "all", False),
    "K5": (5, 0, "all", False),
    "K10": (10, 0, "all", False),
    "K20": (20, 0, "all", False),
    "K50": (50, 0, "all", False),
    "first5": (0, 5, "all", False),
    "first20": (0, 20, "all", False),
    "K10_posonly": (10, 0, "all", True),
    "K10_heavy": (10, 0, "heavy", False),
    "K20_heavy": (20, 0, "heavy", False),
    "K10_star": (10, 0, "star", False),
}


def energy(p, v, m):
    d = p[None] - p[:, None]
    r = torch.sqrt((d ** 2).sum(-1) + EPS ** 2)
    pair = m[:, None] * m[None] / r
    pair.fill_diagonal_(0.0)
    return float(0.5 * (m * (v ** 2).sum(-1)).sum() - 0.5 * pair.sum())


def obs_masks(m, kind, g):
    n = len(m)
    star = torch.zeros(n, dtype=torch.bool, device=DEV)
    star[int(m.argmax())] = True
    if kind is not None:
        heavy = torch.tensor(kind <= 1, device=DEV)
    else:
        order = torch.argsort(-m + 1e-6 * torch.rand(n, generator=g, device=DEV, dtype=torch.float64))
        heavy = torch.zeros(n, dtype=torch.bool, device=DEV)
        heavy[order[: max(1, int(np.ceil(0.1 * n)))]] = True
    return {"all": torch.ones(n, dtype=torch.bool, device=DEV), "heavy": heavy, "star": star}


def roll(net, sc, steps, arm, masks):
    K, first, which, pos_only = ARMS[arm]
    sel = masks[which]
    p, v, m = sc["P"][0], sc["V"][0], sc["m"]
    field = net.init()
    a_prev, field = net.force(sc, p, v, field)
    Pm, Vm = [p], [v]
    for t in range(steps):
        if t > 0 and ((K and t % K == 0) or (first and t < first)):
            p = torch.where(sel[:, None], sc["P"][t], p)
            if not pos_only:
                v = torch.where(sel[:, None], sc["V"][t], v)
            a_prev, _ = net.force(sc, p, v, field)
        pn = p + DT * v + 0.5 * DT ** 2 * a_prev
        vp = v + DT * a_prev
        an, field = net.force(sc, pn, vp, field)
        v = v + 0.5 * DT * (a_prev + an)
        p, a_prev = pn, an
        Pm.append(p)
        Vm.append(v)
    return torch.stack(Pm), torch.stack(Vm)


def window_err(P, sc, ticks, lo, hi):
    if hi > ticks:
        return None
    e = (P[lo:hi + 1] - sc["P"][lo:hi + 1]).norm(dim=-1).mean(1)
    return float(e.mean())


def make_scenes(fam, num, T):
    out = []
    if fam == "solar":
        scene = so.solar_scene(SEED, 200, 4.0)
        Pt, Vt = ss.nbody_truth(scene["pos"], scene["vel"], scene["mass"], T, DEV)
        out.append(dict(P=torch.tensor(Pt, device=DEV), V=torch.tensor(Vt, device=DEV), m=torch.tensor(scene["mass"], dtype=torch.float64, device=DEV), scene=scene))
        return out
    if fam == "star":
        data = sf.make_star_dataset(num, 1, 9200, DEV, dt=DT, eps=EPS)
    else:
        data = gs.make_dataset(num, (100, 100), 1, 9300, scale=True, device=DEV, dt=DT, eps=EPS)
    for d in data:
        p = torch.tensor(d[0][0] - sf.CENTER, dtype=torch.float64, device=DEV)
        v = torch.tensor(d[1][0], dtype=torch.float64, device=DEV)
        m = torch.tensor(d[2], dtype=torch.float64, device=DEV) if len(d) > 2 else torch.ones(len(p), dtype=torch.float64, device=DEV)
        P, V = exact_traj(p, v, m, T)
        out.append(dict(P=P, V=V, m=m))
    return out


def run_family(net, fam, num, T, log):
    scenes = make_scenes(fam, num, T)
    g = torch.Generator(device=DEV)
    g.manual_seed(SEED)
    res = {"n_scenes": len(scenes), "T": T, "N": [len(s["m"]) for s in scenes], "arms": {}}
    for arm in ARMS:
        t0 = time.time()
        wins = {k: [] for k in WINDOWS}
        de, extra = [], []
        for sc in scenes:
            kind = sc["scene"]["kind"] if "scene" in sc else None
            masks = obs_masks(sc["m"], kind, g)
            P, V = roll(net, sc, T, arm, masks)
            for k, (lo, hi) in WINDOWS.items():
                w = window_err(P, sc, T, lo, hi)
                if w is not None:
                    wins[k].append(w)
            fin = bool(torch.isfinite(P).all())
            E0 = abs(energy(sc["P"][0], sc["V"][0], sc["m"]))
            tail = range(max(1, T - 49), T)
            dm = float(np.mean([abs(energy(P[t], V[t], sc["m"]) - energy(sc["P"][t], sc["V"][t], sc["m"])) / E0 for t in tail if t % 50]))
            de.append(dm)
            if fam == "solar":
                extra.append(so.solar_metrics(P.cpu().numpy(), V.cpu().numpy(), sc["P"].cpu().numpy(), sc["V"].cpu().numpy(), sc["scene"], [100, 300, 1000]))
        r = {"err_win": {k: (float(np.mean(v)) if v else None) for k, v in wins.items()}, "err_win_median_scene": {k: (float(np.median(v)) if v else None) for k, v in wins.items()},
             "dE_vs_truth_last50": float(np.mean(de)), "obs_per_body_tick": None}
        K, first, which, _ = ARMS[arm]
        frac = float(masks[which].double().mean())
        r["obs_per_body_tick"] = frac * (1.0 / K if K else (first / T if first else 0.0))
        if extra:
            r["solar"] = extra[0]
        res["arms"][arm] = r
        log(f"{fam} {arm} {json.dumps(r['err_win'])} dE {r['dE_vs_truth_last50']:.4g} ({time.time() - t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fams", default="solar,star,nr100")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    model = ss.build_model("E", DT, DEV, args.ckpt)
    net = Net(model)
    cfg = {"solar": (1, 1000), "star": (12, 150), "nr100": (4, 1000)}
    out = {"ckpt": args.ckpt, "arms": {k: list(v) for k, v in ARMS.items()}}
    log = lambda s: print(s, flush=True)
    with torch.no_grad():
        for fam in args.fams.split(","):
            n, T = cfg[fam]
            if args.smoke:
                n, T = min(n, 2), 120
            out[fam] = run_family(net, fam, n, T, log)
            json.dump(out, open(args.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
