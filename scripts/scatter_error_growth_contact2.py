"""Causal follow-ups for the option-C runaway (diagnostic interventions only, nothing is a proposed fix). See docs/debugging/error-growth-diagnostic.md.
 (a) C weights on gravity (expB) scenes, geometry-free force: does the recurrent potential channel also grow there?  (scene vs weights)
 (b) C on c_id contact scenes with the recurrent potential zeroed every K ticks (K in 5, 10, 20) or clamped to +-CL after each tick.
  PYTHONPATH=. python scripts/scatter_error_growth_contact2.py --ckpt checkpoints_contact/C.pt --out results/errgrowth/contact2.json
"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_contact_scenes as cs
from scripts import scatter_error_growth as eg
from scripts import scatter_field as sf
from scripts import scatter_scenes as ss


class NetC(eg.Net):
    def force(self, sc, p, v, field):
        a, f = sf.ScatterField._force(self.model, p[None].float(), v[None].float(), sc["m"][None].float(), torch.ones(1, len(p), device=eg.DEV), field)
        return a[0].double(), f


def contact_run(model, es, args, dev, reset_k=0, clamp=0.0):
    out = []
    store = {}
    h = model.net.register_forward_hook(lambda mod, inp, o: store.update(x=inp[0].detach(), out=o.detach()))
    for si, s in enumerate(es["scenes"][: args.scenes]):
        n = len(s["mass"])
        pos = torch.tensor(s["pos"], dtype=torch.float32, device=dev)[None]
        vel = torch.tensor(s["vel"], dtype=torch.float32, device=dev)[None]
        mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
        geom = cs._geom_stack([s], dev)
        model.set_scene(geom, cs._rad_pad([s], n).to(dev))
        field = model.init_field(1, dev)
        P = [pos[0].cpu().numpy()]
        tmax, fmax = [], []
        for t in range(args.ticks):
            if reset_k and t > 0 and t % reset_k == 0:
                field = (torch.zeros_like(field[0]), field[1]) if isinstance(field, tuple) else torch.zeros_like(field)
            if clamp:
                field = (field[0].clamp(-clamp, clamp), field[1]) if isinstance(field, tuple) else field.clamp(-clamp, clamp)
            f0 = field[0] if isinstance(field, tuple) else field
            tot, _ = model._force(pos, vel, mass, mask, f0)
            tmax.append(float(tot.norm(dim=-1).max()))
            fmax.append(float(store["x"][0, 5].abs().max()))
            pos, vel, _, field, _ = sf._step_m(model, pos, vel, mass, mask, field)
            P.append(pos[0].cpu().numpy())
        P = np.stack(P)
        err = np.linalg.norm(P[:, :n] - es["Pt"][si][: P.shape[0], :n], axis=-1).mean(1)
        out.append(dict(tot_max=tmax, fld_max=fmax, err=[float(err[k]) for k in (5, 10, 20, 40, args.ticks)]))
    h.remove()
    return out


def summ(rows):
    tm = np.array([max(r["tot_max"][21:]) for r in rows])
    return dict(frac_tot_max_gt_200=float((tm > 200).mean()), median_tot_max_21plus=float(np.median(tm)), max_tot_max_21plus=float(tm.max()),
                err_5_10_20_40_end_mean=[float(np.mean([r["err"][i] for r in rows])) for i in range(5)],
                fld_max_t10_t30_t55_median=[float(np.median([r["fld_max"][t] for r in rows])) for t in (10, 30, 55)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--ticks", type=int, default=60)
    args = ap.parse_args()
    dev = eg.DEV
    model = cs.build("C", dev, args.ckpt)
    res = {}
    with torch.no_grad():
        sc = eg.make_scenes("expB", 8, 9000)
        net = NetC(model)
        model.net.geo = None
        model.momfix = True
        rows = []
        for s in sc:
            rec = []
            Pm, _ = eg.roll(net, s, rec=rec)
            rows.append(dict(fld=[r["fld_max"] for r in rec], a=[r["a_max"] for r in rec], gacc=[r["gacc_max"] for r in rec], ep=(Pm - s["P"]).norm(dim=-1).mean(1).cpu().tolist()))
        res["C_on_gravity_expB"] = dict(fld_max_median_by_tick={t: float(np.median([r["fld"][t - 1] for r in rows])) for t in (1, 5, 10, 20, 30, 50, 100, 150)},
                                       fld_max_max={t: float(max(r["fld"][t - 1] for r in rows)) for t in (1, 5, 10, 20, 30, 50, 100, 150)},
                                       a_max_max={t: float(max(r["a"][t - 1] for r in rows)) for t in (1, 20, 50, 100, 150)},
                                       ep_mean={t: float(np.mean([r["ep"][t] for r in rows])) for t in (5, 10, 20, 50, 100, 150)})
        print("C on gravity", json.dumps(res["C_on_gravity_expB"]), flush=True)
        es = cs.evalset("c_id", ss.EVAL_SEEDS[0], 100, dev)
        for name, kw in (("base", {}), ("fieldreset5", dict(reset_k=5)), ("fieldreset10", dict(reset_k=10)), ("fieldreset20", dict(reset_k=20)),
                         ("clamp100", dict(clamp=100.0)), ("clamp1000", dict(clamp=1000.0))):
            r = contact_run(model, es, args, dev, **kw)
            res["contact_" + name] = summ(r)
            print(name, json.dumps(res["contact_" + name]), flush=True)
    json.dump(res, open(args.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
