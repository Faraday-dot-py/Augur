"""Unit checks for the opt-in scatter-field fixes (far-field multipole + edge taper, substeps, long-unroll penalties). Run on a GPU node:
PYTHONPATH=. python scripts/scatter_fix_tests.py [--ckpt <ckpt>]"""
import argparse

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf


def exact_acc(pos, mass, idx, eps=0.5):
    d = pos - pos[idx]
    r2 = (d ** 2).sum(-1) + eps ** 2
    w = mass * r2 ** -1.5
    w[idx] = 0
    return (w[:, None] * d).sum(0)


def far_multipole(dev):
    g = torch.Generator().manual_seed(4738)
    cl = (torch.randn(60, 2, generator=g) * 4.0).to(dev)
    m = (0.5 + torch.rand(60, generator=g)).to(dev)
    model = sf.ScatterField(grid=64, extent=64.0, net="unet", potential=True, verlet=True).to(dev)
    rows = []
    for R in (20.0, 30.0, 40.0, 60.0):
        pos = torch.cat([cl, torch.tensor([[R, 3.0]], device=dev)])[None]
        mass = torch.cat([m, torch.tensor([0.01], device=dev)])[None]
        mask = torch.ones(1, 61, device=dev)
        ex = exact_acc(pos[0], mass[0], 60)
        errs = []
        for far in (1, 2):
            model.far = far
            a = model.far_acc(pos, mass, mask)[0, 60]
            errs.append(float((a - ex).norm() / ex.norm()))
        rows.append((R, errs))
        print(f"far-field R={R}: |exact| {float(ex.norm()):.3e} rel err mono {errs[0]:.2e} quad {errs[1]:.2e}")
    for R, (e1, e2) in rows:
        assert e1 < 0.2 and e2 < e1 * 1.01, (R, e1, e2)
    pos = torch.tensor([[[0.0, 0.0], [16.0, 0.0], [24.0, 0.0], [31.9, 0.0], [40.0, 0.0], [0.0, -28.0]]], device=dev)
    w = model.edge_weight(pos)[0].tolist()
    print("edge weights (band .25):", [round(x, 3) for x in w])
    assert w[0] == 1.0 and w[1] == 1.0 and w[2] == 1.0 and w[3] < 0.01 and w[4] == 0.0 and 0 < w[5] < 1


def flags_off_identity(dev, ckpt):
    model = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()
    g = torch.Generator().manual_seed(4738)
    pos = ((torch.rand(2, 50, 2, generator=g) - 0.5) * 20).to(dev)
    vel = torch.randn(2, 50, 2, generator=g).to(dev)
    mass, mask = torch.ones(2, 50, device=dev), torch.ones(2, 50, device=dev)
    with torch.no_grad():
        ref = model.step(pos, vel, mass, mask, model.init_field(2, dev))
        sf.apply_opt(model, "far")
        far_in = model.step(pos, vel, mass, mask, model.init_field(2, dev))
        model.far = 0
        model.substeps = 2
        sub = model.step(pos, vel, mass, mask, model.init_field(2, dev))
        model.substeps = 1
        again = model.step(pos, vel, mass, mask, model.init_field(2, dev))
    assert torch.equal(ref[0], again[0]) and torch.equal(ref[1], again[1])
    d_far = float((ref[0] - far_in[0]).abs().max())
    d_sub = float((ref[0] - sub[0]).abs().max())
    print(f"flags off identical; far on, scene inside 0.75 of the box: max pos diff {d_far:.3e}; substeps=2 max pos diff {d_sub:.3e}")
    assert d_far == 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    far_multipole(dev)
    flags_off_identity(dev, args.ckpt)
    print("ALL OK")


if __name__ == "__main__":
    main()
