"""Blow-up diagnosis for a trained option-C model: per tick, max |a| of each force term and the geometry of the worst ball.

  python scripts/scatter_contact_c_diag.py --ckpt checkpoints_contact/C.pt --fam c_id [--scene 0] [--ticks 60]
Terms: a0 (field net without geometry + pair prior, momfix off), geo (a_geo - a0, before the FMAX bound), pair (learned pair head), sdf/gap of the ball with the largest |v|.
"""
import argparse

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_contact/C.pt")
    ap.add_argument("--fam", default="c_id")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=60)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = cs.build("C", dev, args.ckpt)
    for o in ("kcache", "fastio"):
        sf.apply_opt(model, o)
    es = cs.evalset(args.fam, ss.EVAL_SEEDS[0], 100, dev)
    s = es["scenes"][args.scene]
    n = len(s["mass"])
    pos = torch.tensor(s["pos"], dtype=torch.float32, device=dev)[None]
    vel = torch.tensor(s["vel"], dtype=torch.float32, device=dev)[None]
    mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
    geom = cs._geom_stack([s], dev)
    rad = cs._rad_pad([s], n).to(dev)
    model.set_scene(geom, rad)
    field = model.init_field(1, dev)
    print("tick  max|v|  worst ball: sdf  nearest-ball gap | max|a0| max|geo| max|pair|  | max|a_total|", flush=True)
    with torch.no_grad():
        for t in range(args.ticks):
            f0 = field[0] if isinstance(field, tuple) else field
            mf = model.momfix
            model.momfix = False
            model.net.geo = sc.chan_feats(geom, model.h)
            a_geo, _ = sf.ScatterField._force(model, pos, vel, mass, mask, f0)
            model.net.geo = None
            a0, _ = sf.ScatterField._force(model, pos, vel, mass, mask, f0)
            model.momfix = mf
            pair = model.pair_acc(pos, mask, rad)
            i = int(vel[0].norm(dim=-1).argmax())
            sd = float(sc.scene_sdf(s["prims"], pos[0, i].double().cpu()))
            d = (pos[0] - pos[0, i]).norm(dim=-1)
            d[i] = 1e9
            gap = float(d.min()) - 2 * float(rad[0, i])
            tot, _ = model._force(pos, vel, mass, mask, f0)
            print(f"{t:3d} {float(vel.norm(dim=-1).max()):8.2f}  {sd:7.2f} {gap:7.2f} | {float(a0.norm(dim=-1).max()):8.2f} {float((a_geo - a0).norm(dim=-1).max()):8.2f} {float(pair.norm(dim=-1).max()):8.2f} | {float(tot.norm(dim=-1).max()):8.2f}", flush=True)
            pos, vel, _, field, _ = sf._step_m(model, pos, vel, mass, mask, field)


main()
