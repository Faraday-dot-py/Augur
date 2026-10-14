"""Is the no-geometry path of a frozen-branch checkpoint equal to Exp E on large scenes, and how large is E's own run-to-run GPU nondeterminism?
Usage: python scripts/bfrozen_noise.py BF_CKPT [E_CKPT]"""
import sys

import numpy as np
import torch

from scripts import scatter_contact_scenes as cs
from scripts import scatter_field as sf
from scripts import train_scatter_field as tsf

dev = torch.device("cuda")
BF = sys.argv[1]
E = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"


def forces(model, pos, vel, mass, mask, ticks=10):
    field = model.init_field(pos.shape[0], dev)
    out = []
    with torch.no_grad():
        for _ in range(ticks):
            pos, vel, field, _ = model.step(pos, vel, mass, mask, field)
        a, _ = model.force(pos, vel, mass, mask, field)
    return a, pos


def main():
    me = tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", 0.1).to(dev)
    sf.load_compat(me, torch.load(E, map_location=dev, weights_only=False)["model"])
    me.eval()
    mb = cs.build("Bf", dev, BF)
    mb.set_scene(None, None)
    mb.dt = 0.1
    rng = np.random.default_rng(4738)
    for n in (10, 100, 300):
        pos = torch.tensor(rng.normal(0, 8, (1, n, 2)), dtype=torch.float32, device=dev)
        vel = torch.tensor(rng.normal(0, 1, (1, n, 2)), dtype=torch.float32, device=dev)
        mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
        a1, p1 = forces(me, pos, vel, mass, mask)
        a2, p2 = forces(me, pos, vel, mass, mask)
        a3, p3 = forces(mb, pos, vel, mass, mask)
        a4, p4 = forces(mb, pos, vel, mass, mask)
        print(f"N {n}: E vs E rerun max|da| {float((a1 - a2).abs().max()):.3e} pos {float((p1 - p2).abs().max()):.3e} | Bf vs E max|da| {float((a3 - a1).abs().max()):.3e} pos {float((p3 - p1).abs().max()):.3e} | Bf vs Bf rerun {float((a3 - a4).abs().max()):.3e} | |a| max {float(a1.abs().max()):.3g}", flush=True)


main()
