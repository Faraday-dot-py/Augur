"""Property test of the energy-similarity loss term (train_scatter_field.energy_sim): it must score similarity to the truth energy trajectory, not conservation.
Truth = exact softened gravity (eps 0.5, unit masses, KDK, dt 0.1, 4 substeps) with a per-tick velocity factor f: f<1 drag (energy drains), f>1 injection, f=1 conserving.
'tracker' = truth trajectory + small position/velocity noise; 'conserver' = the same initial state integrated with f=1.
Scores both with energy_sim (term used in training, k=20 window averaged as in train) and with a conservation penalty |E(t)-E(0)|/scale for contrast.
Pass: tracker esim << conserver esim whenever truth drifts; on f=1 both ~equal and small.
Usage: PYTHONPATH=. python scripts/scatter_esim_test.py"""
import json

import numpy as np
import torch

from scripts import gravity_sim as gs
from scripts import train_scatter_field as ts

EPS, DT, SUBS, K = 0.5, 0.1, 4, 20


def acc(p):
    d = p[:, None] - p[:, :, None]
    inv = ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
    inv.diagonal(dim1=1, dim2=2).zero_()
    return (d * inv[..., None]).sum(2)


def simulate(p, v, f):
    P, V = [p], [v]
    h = DT / SUBS
    a = acc(p)
    for _ in range(K):
        for _ in range(SUBS):
            v = v + 0.5 * h * a
            p = p + h * v
            a = acc(p)
            v = v + 0.5 * h * a
        v = v * f
        P.append(p)
        V.append(v)
    return torch.stack(P, 1), torch.stack(V, 1)


def score(Pm, Vm, Pt, Vt, m, mk):
    sc = ts.energy_scale(Pt[:, 0], Vt[:, 0], m, mk, EPS)
    es = sum(ts.energy_sim(Pm[:, s], Vm[:, s], m, mk, Pt[:, s], Vt[:, s], m, EPS, sc) for s in range(1, K + 1)) / K
    sg = sum(ts.energy_sim(Pm[:, s], Vm[:, s], m, mk, Pt[:, s], Vt[:, s], m, EPS, sc, signed=True) for s in range(1, K + 1)) / K
    E0 = ts.sf.energy(Pm[:, 0], Vm[:, 0], m, mk, EPS)
    cons = sum(((ts.sf.energy(Pm[:, s], Vm[:, s], m, mk, EPS) - E0).abs() / sc).mean() for s in range(1, K + 1)) / K
    pos = sum((((Pm[:, s] - Pt[:, s]) ** 2).sum(-1).sqrt()).mean() for s in range(1, K + 1)) / K
    return {"esim": float(es), "esim_signed": float(sg), "conservation_pen": float(cons), "pos_err": float(pos)}


def main():
    torch.manual_seed(4738)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(4738)
    pv = [gs.init_bodies(20, rng, scale=True) for _ in range(8)]
    p0 = torch.tensor(np.stack([p for p, _ in pv]) - gs.CENTER, dtype=torch.float64, device=dev)
    v0 = torch.tensor(np.stack([v for _, v in pv]), dtype=torch.float64, device=dev)
    m = torch.ones(p0.shape[:2], dtype=torch.float64, device=dev)
    mk = torch.ones_like(m)
    out = {}
    Pc, Vc = simulate(p0, v0, 1.0)
    for name, f in (("drag_f0.98", 0.98), ("drag_f0.995", 0.995), ("conserving_f1", 1.0), ("inject_f1.01", 1.01)):
        Pt, Vt = simulate(p0, v0, f)
        Ptr = Pt + 1e-3 * torch.randn_like(Pt)
        Vtr = Vt + 1e-3 * torch.randn_like(Vt)
        Ptr[:, 0], Vtr[:, 0] = Pt[:, 0], Vt[:, 0]
        sc = ts.energy_scale(Pt[:, 0], Vt[:, 0], m, mk, EPS)
        dEt = float(((ts.sf.energy(Pt[:, -1], Vt[:, -1], m, mk, EPS) - ts.sf.energy(Pt[:, 0], Vt[:, 0], m, mk, EPS)) / sc).mean())
        out[name] = {"truth_dE_over_scale_at_k": dEt, "tracker": score(Ptr, Vtr, Pt, Vt, m, mk), "conserver": score(Pc, Vc, Pt, Vt, m, mk)}
        r = out[name]
        print(f"{name}: truth dE/scale@{K} {dEt:+.4f} | esim tracker {r['tracker']['esim']:.5f} conserver {r['conserver']['esim']:.5f} "
              f"| conservation-penalty tracker {r['tracker']['conservation_pen']:.5f} conserver {r['conserver']['conservation_pen']:.5f}", flush=True)
    ok = all(out[n]["tracker"]["esim"] < 0.1 * out[n]["conserver"]["esim"] for n in ("drag_f0.98", "drag_f0.995", "inject_f1.01"))
    ok = ok and out["conserving_f1"]["conserver"]["esim"] < 1e-12 and out["conserving_f1"]["tracker"]["esim"] < 0.01
    out["pass"] = ok
    print("PASS" if ok else "FAIL", flush=True)
    json.dump(out, open("results/esim_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
