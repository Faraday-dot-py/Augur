"""Second energy probe: controlled wall / pair contact curves vs incoming speed
(truth vs soup B), truth total-energy conservation at high speed, training-set
speed distribution, hidden-zeroed rollout.

Usage: PYTHONPATH=. python3 scripts/energy_probe2.py
"""
import argparse
import json
import random

import numpy as np
import torch

from model.dataset import make_scenario_uniform
from model.token_dataset import load_dataset_samples
from scripts.ball_1k_rollout import forces, truth
from scripts.energy_probe import energy
from scripts.eval_free_rollout import load_model

DT, RADIUS, K = 0.15, 0.75, 400.0
SPEEDS = [3, 6, 10, 15, 20, 30, 40, 60, 80]
G = 9.0


def truth_states(p0, v0, n, steps, device, g=G):
    p = torch.tensor(p0, dtype=torch.float64, device=device)
    v = torch.tensor(v0, dtype=torch.float64, device=device)
    ps, vs = [p.float().cpu().numpy()], [v.float().cpu().numpy()]
    sub = DT / 8
    for _ in range(steps):
        for _ in range(8):
            v = v + forces(p, n, g, RADIUS, K) * sub
            p = p + v * sub
        ps.append(p.float().cpu().numpy())
        vs.append(v.float().cpu().numpy())
    return np.array(ps), np.array(vs)


def model_states(model, p0, v0, steps, device):
    pos, vel = torch.tensor(p0, dtype=torch.float32, device=device), torch.tensor(v0, dtype=torch.float32, device=device)
    hid = torch.zeros(pos.shape[0], model.dynamics.hidden_dim, device=device)
    ps, vs = [pos.cpu().numpy()], [vel.cpu().numpy()]
    with torch.no_grad():
        for _ in range(steps):
            pos, vel, hid, _ = model.step_free(pos, vel, hid, render=False)
            ps.append(pos.cpu().numpy())
            vs.append(vel.cpu().numpy())
    return np.array(ps), np.array(vs)


def wall_curve(model, device, n=800):
    rows = []
    for axis, name in ((1, "y_wall"), (0, "x_floor")):
        for s in SPEEDS:
            p0 = [[400.0, 400.0]]
            v0 = [[0.0, 0.0]]
            if axis == 1:
                p0[0][1] = 8.0
                v0[0][1] = -float(s)
            else:
                p0[0][0] = (n - 1.0) - 8.0
                v0[0][0] = float(s)
            steps = 40
            row = {"wall": name, "speed_in": s}
            for label, (ps, vs) in (("truth", truth_states(p0, v0, n, steps, device)),
                                    ("model", model_states(model, p0, v0, steps, device))):
                if axis == 1:
                    vin = abs(vs[0][0][1])
                    vout = float(vs[-1][0][1])
                    row[label + "_vy_after"] = vout
                    row[label + "_ratio"] = vout / vin
                    row[label + "_ke_ratio"] = float(0.5 * (vs[-1][0] ** 2).sum()) / float(0.5 * (vs[0][0] ** 2).sum())
                else:
                    e0 = float(energy(torch.tensor(ps[0]), torch.tensor(vs[0]), G)[0])
                    e1 = float(energy(torch.tensor(ps[-1]), torch.tensor(vs[-1]), G)[0])
                    row[label + "_vx_after"] = float(vs[-1][0][0])
                    row[label + "_dE"] = e1 - e0
                    row[label + "_E0"] = e0
            rows.append(row)
    return rows


def pair_curve(model, device, n=800):
    rows = []
    for off in (0.0, 0.6):
        for s in SPEEDS:
            p0 = [[400.0, 396.0], [400.0 + off, 404.0]]
            v0 = [[0.0, float(s)], [0.0, -float(s)]]
            steps = 30
            row = {"offset": off, "speed_in": s}
            for label, (ps, vs) in (("truth", truth_states(p0, v0, n, steps, device)),
                                    ("model", model_states(model, p0, v0, steps, device))):
                rel0 = vs[0][1] - vs[0][0]
                rel1 = vs[-1][1] - vs[-1][0]
                row[label + "_relKE_ratio"] = float((rel1 ** 2).sum() / (rel0 ** 2).sum())
                row[label + "_pmom_err"] = float(np.abs((vs[-1].sum(0) - vs[0].sum(0)) - np.array([2 * G * DT * steps, 0.0])).max())
                row[label + "_finite"] = bool(np.isfinite(vs).all())
            rows.append(row)
    return rows


def spring_pe(p, n):
    lo, hi = 0.0, n - 1.0
    pen_w = torch.cat([(RADIUS - (p - lo)).clamp(min=0), (RADIUS - (hi - p)).clamp(min=0)], dim=1)
    pe = (K * pen_w ** 3 / 3).sum(1)
    d = (p[:, None] - p[None]).norm(dim=-1)
    pen_p = (2 * RADIUS - d).clamp(min=0)
    pen_p.fill_diagonal_(0.0)
    return pe + (K * pen_p ** 3 / 3).sum(1) / 2


def truth_conservation(device, nb=1000, n=317, steps=100):
    balls = make_scenario_uniform(nb, n, 2.3, random.Random(4738))
    p = torch.tensor([[b["x"], b["y"]] for b in balls], dtype=torch.float64, device=device)
    v = torch.tensor([[b["vx"], b["vy"]] for b in balls], dtype=torch.float64, device=device)
    sub = DT / 8
    rows = []
    for t in range(steps + 1):
        tot = (0.5 * (v ** 2).sum(-1) - G * p[:, 0] + spring_pe(p, n)).mean()
        rows.append({"t": t, "E_total": float(tot), "KE": float((0.5 * (v ** 2).sum(-1)).mean()),
                     "E_nospring": float((0.5 * (v ** 2).sum(-1) - G * p[:, 0]).mean()), "maxspeed": float(v.norm(dim=1).max())})
        for _ in range(8):
            v = v + forces(p, n, G, RADIUS, K) * sub
            p = p + v * sub
    return rows


def dataset_speeds(path):
    samples = load_dataset_samples(path)
    sp = []
    for _, states in samples:
        for fr in states:
            sp.extend([(b["vx"] ** 2 + b["vy"] ** 2) ** 0.5 for b in fr])
    sp = np.array(sp)
    return {"n": int(len(sp)), "mean": float(sp.mean()), "p50": float(np.percentile(sp, 50)), "p90": float(np.percentile(sp, 90)),
            "p99": float(np.percentile(sp, 99)), "p999": float(np.percentile(sp, 99.9)), "max": float(sp.max()),
            "frac_gt10": float((sp > 10).mean()), "frac_gt15": float((sp > 15).mean()), "frac_gt20": float((sp > 20).mean())}


def hidden_reset_rollout(ckpt, device, steps=100, seeds=8):
    n, nb = 20, 4
    model = load_model(ckpt, "free", n, 32, 4.0, False, True, True, True, True, True, True).to(device)
    hd = model.dynamics.hidden_dim
    out = {"carried": [], "zeroed": [], "truth": []}
    for i in range(seeds):
        balls = make_scenario_uniform(nb, n, 2.3, random.Random(4738 + i))
        tp, tv = truth(balls, n, steps, DT, G, RADIUS, K, 8, device)
        out["truth"].append([float(energy(torch.tensor(tp[t]), torch.tensor(tv[t]), G).mean()) for t in range(steps + 1)])
        for mode in ("carried", "zeroed"):
            pos, vel = torch.tensor(tp[0], device=device), torch.tensor(tv[0], device=device)
            hid = torch.zeros(nb, hd, device=device)
            es, errs = [], []
            with torch.no_grad():
                for t in range(steps + 1):
                    es.append(float(energy(pos, vel, G).mean()))
                    errs.append(float((pos - torch.tensor(tp[t], device=device)).norm(dim=1).mean()))
                    pos, vel, hid, _ = model.step_free(pos, vel, torch.zeros_like(hid) if mode == "zeroed" else hid, render=False)
            out[mode].append(es)
            out[mode + "_err"] = out.get(mode + "_err", []) + [errs]
    res = {}
    for k in ("carried", "zeroed", "truth"):
        a = np.array(out[k]).mean(0)
        res[k] = {str(t): float(a[t]) for t in (0, 5, 10, 20, 30, 50, 75, 100)}
    for k in ("carried_err", "zeroed_err"):
        a = np.array(out[k]).mean(0)
        res[k] = {str(t): float(a[t]) for t in (1, 5, 10, 20, 30, 50, 100)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--dataset", default="checkpoints/token_dataset_6000_h44_seed4738.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/energy_probe2.json")
    args = ap.parse_args()
    dev = args.device
    model = load_model(args.checkpoint, "free", 800, 32, 4.0, False, True, True, True, True, True, True).to(dev)
    model.dynamics.cell_graph = False
    res = {}
    res["dataset_speeds"] = dataset_speeds(args.dataset)
    print("dataset speeds", res["dataset_speeds"], flush=True)
    res["truth_conservation"] = truth_conservation(dev)
    for r in res["truth_conservation"][::10]:
        print("truth cons", r, flush=True)
    res["wall_curve"] = wall_curve(model, dev)
    for r in res["wall_curve"]:
        print("wall", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res["pair_curve"] = pair_curve(model, dev)
    for r in res["pair_curve"]:
        print("pair", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res["hidden_reset"] = hidden_reset_rollout(args.checkpoint, dev)
    print("hidden_reset", json.dumps(res["hidden_reset"]), flush=True)
    json.dump(res, open(args.out, "w"), default=float)
    print("done")


if __name__ == "__main__":
    main()
