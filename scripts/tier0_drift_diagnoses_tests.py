"""Tests for the pure helpers of scripts/tier0_drift_diagnoses.py (and the probe plumbing with the exact force): PYTHONPATH=. python scripts/tier0_drift_diagnoses_tests.py [--device cuda]"""
import argparse

import numpy as np
import torch

from scripts import tier0_drift_diagnoses as dg
from scripts import train_tiered as ttr

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not ok:
        fails.append(name)


def ring(R, K=64):
    th = torch.arange(K, dtype=torch.float64) * (2 * np.pi / K)
    return R * torch.stack([th.cos(), th.sin()], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    for R in (1.0, 4.0):
        x = ring(R)
        w, n = dg.ring_work(-x / x.norm(dim=-1, keepdim=True) ** 3, x)
        check(f"conservative central force has zero loop work R={R}", abs(float(w / n)) < 1e-9, f"{float(w / n):.2e}")
        w, n = dg.ring_work(torch.stack([-x[:, 1], x[:, 0]], -1), x)
        check(f"rotational field loop-work ratio is 1 R={R}", abs(float(w / n) - 1.0) < 1e-6, f"{float(w / n):.6f}")
    g = torch.Generator().manual_seed(4738)
    v, at, u = (torch.randn(4000, 2, generator=g, dtype=torch.float64) for _ in range(3))
    e = -0.3 * v + 0.05 * at + 0.2 * u
    r = dg.regress_force_error(e, v, at, u)
    check("regression recovers drag, gain and inward coefficients", abs(r["coef_drag"] - 0.3) < 1e-9 and abs(r["coef_gain"] - 0.05) < 1e-9 and abs(r["coef_inward"] - 0.2) < 1e-9)
    check("regression work shares sum to one", abs(r["work_share_drag"] + r["work_share_gain"] + r["work_share_inward"] - 1.0) < 1e-9)
    e = -0.3 * v
    r = dg.regress_force_error(e, v, at, u)
    check("pure drag error gives drag work share 1", abs(r["work_share_drag"] - 1.0) < 1e-6, f"{r['work_share_drag']:.6f}")
    check("slope of a line", abs(dg.slope(torch.arange(21, dtype=torch.float64) * -0.002) + 0.002) < 1e-12)
    S, N = 3, 100
    gen = torch.Generator().manual_seed(4738)
    pos = torch.zeros(S, N, 2, dtype=torch.float64)
    pos[:, :12] = torch.randn(S, 12, 2, generator=gen, dtype=torch.float64) * 2
    mask = torch.zeros(S, N, dtype=torch.float64)
    mask[:, :12] = 1
    mass = mask.clone()
    vel = torch.zeros_like(pos)
    rg = ring(2.0).to(dev)[None].expand(S, -1, -1)
    pa, ta = dg.probe_force(lambda p, v_, m, k, f: ttr.true_acc(p.float(), m.float(), k.float()), pos.float().to(dev), vel.float().to(dev), mass.float().to(dev), mask.float().to(dev), None, rg)
    check("probe force equals exact force at the ring", float((pa - ta).abs().max()) < 1e-6)
    w, n = dg.ring_work(ta.double(), rg.double())
    check("exact-force control loop work is small", float((w / n).abs().max()) < 5e-3, f"{float((w / n).abs().max()):.2e}")
    pe = dg.pe_particle(pos, mask)
    check("per-particle potential sums to the pair potential", abs(float(pe[0].sum()) - float(-0.5 * (1 / torch.sqrt(dg.pair_r(pos, mask)[0][0] ** 2 + dg.EPS ** 2) * dg.pair_r(pos, mask)[1][0]).sum())) < 1e-9)
    sel = dg.densest_third(pos, mask)
    check("densest third selects about a third of the masked particles", all(3 <= int(sel[s].sum()) <= 5 for s in range(S)), str(sel.sum(-1).tolist()))
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
