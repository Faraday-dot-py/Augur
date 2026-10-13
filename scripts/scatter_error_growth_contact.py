"""Input-OOD probe of the option-C runaway (contact C ckpt, E-derived grid net with no geometry channels). See docs/debugging/error-growth-diagnostic.md.

  PYTHONPATH=. python scripts/scatter_error_growth_contact.py --ckpt checkpoints_contact/C.pt --fam c_id --out results/errgrowth/contact_C.json
Per tick, net-input/output stats of the a0 pass (geometry-free, momfix off), force terms, and where the largest grid acceleration sits.
Envelope = ticks 0-20 (training k_end) of the same scenes.
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_contact as sc
from scripts import scatter_contact_scenes as cs
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss

G = 128
ENV_T = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fam", default="c_id")
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = cs.build("C", dev, args.ckpt)
    store = {}
    model.net.register_forward_hook(lambda mod, inp, out: store.update(x=inp[0].detach(), out=out.detach()))
    es = cs.evalset(args.fam, ss.EVAL_SEEDS[0], 100, dev)
    rows = []
    with torch.no_grad():
        for si, s in enumerate(es["scenes"][: args.scenes]):
            n = len(s["mass"])
            pos = torch.tensor(s["pos"], dtype=torch.float32, device=dev)[None]
            vel = torch.tensor(s["vel"], dtype=torch.float32, device=dev)[None]
            mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
            geom = cs._geom_stack([s], dev)
            model.set_scene(geom, cs._rad_pad([s], n).to(dev))
            field = model.init_field(1, dev)
            tr = []
            for t in range(args.ticks):
                f0 = field[0] if isinstance(field, tuple) else field
                mf = model.momfix
                model.momfix = False
                model.net.geo = None
                a0, _ = sf.ScatterField._force(model, pos, vel, mass, mask, f0)
                model.momfix = mf
                x, out = store["x"][0], store["out"][0]
                gn = model.neg_grad(store["out"][:, :1])[0].norm(dim=0)
                iy, ix = divmod(int(gn.argmax()), G)
                cell = torch.tensor([(ix + 0.5) * model.h - 32.0, (iy + 0.5) * model.h - 32.0], device=dev)
                d = (pos[0] - cell).norm(dim=-1)
                model.net.geo = sc.chan_feats(geom, model.h)
                a_geo, _ = sf.ScatterField._force(model, pos, vel, mass, mask, f0)
                model.net.geo = None
                tot, _ = model._force(pos, vel, mass, mask, f0)
                tr.append(dict(rho_max=float(x[0].max()), mom_max=float(x[1:3].abs().max()), ak_max=float(x[3:5].abs().max()), fld_max=float(x[5].abs().max()),
                               phi_max=float(out[0].abs().max()), gacc_max=float(gn.max()), a0_max=float(a0.norm(dim=-1).max()), geo_max=float((a_geo - a0).norm(dim=-1).max()),
                               tot_max=float(tot.norm(dim=-1).max()), vmax=float(vel.norm(dim=-1).max()), cell_x=float(cell[0]), cell_y=float(cell[1]),
                               cell_nearest_ball=float(d.min()), cell_sdf=float(sc.scene_sdf(s["prims"], cell.double().cpu())), pmax=float(pos.abs().max())))
                pos, vel, _, field, _ = sf._step_m(model, pos, vel, mass, mask, field)
            rows.append(tr)
            print(f"scene {si} n={n} a0_max t20/22/24/30/40 {[round(tr[k]['a0_max'], 1) for k in (20, 22, 24, 30, 40) if k < len(tr)]}", flush=True)
    keys = list(rows[0][0].keys())
    env = {k: max(r[t][k] for r in rows for t in range(ENV_T + 1)) for k in keys}
    out = dict(ckpt=args.ckpt, fam=args.fam, env_ticks=ENV_T, env_max=env, scenes=rows)
    first = {}
    for k in keys:
        if k.startswith("cell") or k in ("pmax",):
            continue
        first[k] = [next((t for t in range(len(r)) if r[t][k] > env[k] and t > ENV_T), None) for r in rows]
    out["first_exceed_env_tick_per_scene"] = first
    json.dump(out, open(args.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
