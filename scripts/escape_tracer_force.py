"""Static force on massless tracers outside the cluster: model force vs exact (from the massive bodies), by radius, after rolling the massive bodies 0/20/100 ticks with each arm.
Tracers on rings r = 20..150 (24 angles), velocity 0.95c outward as in the tracer scene. Reports the inward radial force ratio model/exact (median) and tangential rms error / exact
radial, with momfix on and off. Arms as in escape_roll.ARMS.
Usage: PYTHONPATH=. python scripts/escape_tracer_force.py --ckpt X.pt --n 20000 --seed 4738 --arms S1,T4,T8,T8w,T16,T8nf,T8old --out results/escape/tracer_force.json"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_bh as sb
from scripts.escape_roll import Arm, C, DT, EPS, make_ic, mom_of

RADII = [20, 25, 30, 35, 40, 50, 60, 80, 100, 150]
NANG = 24
TICKS = (0, 20, 100)


def exact_on(src, M, tr):
    out = torch.zeros_like(tr)
    for s in range(0, tr.shape[0], 100):
        d = src[None] - tr[s:s + 100, None]
        inv = ((d ** 2).sum(-1) + EPS ** 2) ** -1.5
        out[s:s + 100] = (d * (inv * M[None])[..., None]).sum(1)
    return out


@torch.no_grad()
def run(arm, pos0, vel0, mass0, dev):
    pos = torch.tensor(pos0, dtype=torch.float32, device=dev)
    mom = mom_of(torch.tensor(vel0, dtype=torch.float32, device=dev))
    M = torch.tensor(mass0, dtype=torch.float32, device=dev)
    f = arm.init_field()
    a, f = arm.force(pos, sb.speed(mom, C), M, f)
    ang = torch.arange(NANG, device=dev) * (2 * np.pi / NANG) + 0.1
    rh = torch.stack([ang.cos(), ang.sin()], -1)
    res = {}
    for t in range(max(TICKS) + 1):
        if t in TICKS:
            tr_pos = torch.cat([r * rh for r in RADII]).float()
            tr_hat = rh.repeat(len(RADII), 1)
            tr_r = torch.tensor([r for r in RADII for _ in range(NANG)], device=dev, dtype=torch.float32)
            allp = torch.cat([pos, tr_pos])
            allv = torch.cat([sb.speed(mom, C), 0.95 * C * tr_hat])
            allM = torch.cat([M, torch.zeros(len(tr_pos), device=dev)])
            ex = exact_on(pos.double(), M.double(), tr_pos.double())
            row = {}
            for mf in (True, False):
                old = arm.set_momfix(mf)
                am, _ = arm.force(allp, allv, allM, f)
                arm.set_momfix(old)
                am = am[len(pos):].double()
                rin_m, rin_e = -(am * tr_hat).sum(-1), -(ex * tr_hat).sum(-1)
                tan = torch.stack([-tr_hat[:, 1], tr_hat[:, 0]], -1)
                for r in RADII:
                    s = tr_r == r
                    row.setdefault(f"r{r}", {})["mf" if mf else "nomf"] = {"ratio_med": float((rin_m[s] / rin_e[s]).median()), "exact_in": float(rin_e[s].mean()), "model_in": float(rin_m[s].mean()),
                                                                          "tan_rms": float(((am * tan).sum(-1)[s] ** 2).mean().sqrt())}
            res[t] = row
        if t == max(TICKS):
            break
        mom = mom + 0.5 * DT * a
        pos = pos + DT * sb.speed(mom, C)
        a, f = arm.force(pos, sb.speed(mom, C), M, f)
        mom = mom + 0.5 * DT * a
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--arms", default="S1,T4,T8,T8w,T16,T8nf,T8old")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    pos0, vel0, mass0 = make_ic(args.n, args.seed, 0)
    out = {}
    for name in args.arms.split(","):
        out[name] = run(Arm(name, args.ckpt, dev), pos0, vel0, mass0, dev)
        for t in TICKS:
            print(name, f"t{t} inward ratio med (momfix on / off) by r: " + " ".join(f"{r}:{out[name][t][f'r{r}']['mf']['ratio_med']:.2f}/{out[name][t][f'r{r}']['nomf']['ratio_med']:.2f}" for r in RADII), flush=True)
    json.dump(out, open(args.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
