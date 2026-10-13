"""Softcap on the fed-back field (ScatterField.softcap) for contact-C ckpts: runaway probe on contact (c_id) and gravity (expB, cluster) scenes, fresh seeds.
See docs/debugging/error-correction-softcap-ship.md.

  PYTHONPATH=. python scripts/scatter_softcap_eval.py --ckpt checkpoints_contact/C.pt --softcaps 0,70,100,140 --out results/softcap/C.json
Contact seeds 9400/9500/9600 x 24 scenes, 100 ticks; gravity expB (12 scenes) and cluster (6) per seed, 150 ticks, geometry-free C force. Per scene: max |a| over ticks >= 21
(total, grid part = total - local heads, local = gvec + pair head), max raw field |.| (net output), err@20/@100, off-grid fraction (|x| >= 32) at the last tick.
"""
import argparse
import json
import time

import numpy as np
import torch

from scripts import scatter_contact_scenes as cs
from scripts import scatter_error_growth as eg
from scripts import scatter_error_growth_contact2 as c2
from scripts import scatter_field as sf

SEEDS = (9400, 9500, 9600)
HALF = 32.0


def offgrid(P):
    return float((P.abs().amax(-1) >= HALF).float().mean())


def contact_rows(model, es, ticks, dev):
    rows = []
    store = {}
    h = model.net.register_forward_hook(lambda mod, inp, o: store.update(out=o.detach()))
    for si, s in enumerate(es["scenes"]):
        n = len(s["mass"])
        pos = torch.tensor(s["pos"], dtype=torch.float32, device=dev)[None]
        vel = torch.tensor(s["vel"], dtype=torch.float32, device=dev)[None]
        mass, mask = torch.ones(1, n, device=dev), torch.ones(1, n, device=dev)
        model.set_scene(cs._geom_stack([s], dev), cs._rad_pad([s], n).to(dev))
        field = model.init_field(1, dev)
        P = [pos[0]]
        tot, grid, loc, fld = [], [], [], []
        for t in range(ticks):
            f0 = field[0] if isinstance(field, tuple) else field
            a, _ = model._force(pos, vel, mass, mask, f0)
            l = model.local_acc(pos, vel, mask)
            tot.append(float(a.norm(dim=-1).max()))
            loc.append(float(l.norm(dim=-1).max()))
            grid.append(float((a - l).norm(dim=-1).max()))
            fld.append(float(store["out"][0, 0].abs().max()))
            pos, vel, _, field, _ = sf._step_m(model, pos, vel, mass, mask, field)
            P.append(pos[0])
        P = torch.stack(P)
        err = (P - torch.tensor(es["Pt"][si][: ticks + 1, :n], device=dev, dtype=P.dtype)).norm(dim=-1).mean(1).cpu().numpy()
        rows.append(dict(tot=tot[21:], grid=grid[21:], loc=loc[21:], fld=fld, err20=float(err[20]), err100=float(err[ticks]), off100=offgrid(P[-1]), finite=bool(torch.isfinite(P).all())))
    h.remove()
    return rows


def summ(rows):
    tm = np.array([max(r["tot"]) for r in rows])
    return dict(n=len(rows), runaway=float((tm > 200).mean()), tot_max_median=float(np.median(tm)), tot_max_max=float(tm.max()),
                grid_max_median=float(np.median([max(r["grid"]) for r in rows])), loc_max_median=float(np.median([max(r["loc"]) for r in rows])),
                fld_max_median=float(np.median([max(r["fld"]) for r in rows])), err20=float(np.mean([r["err20"] for r in rows])),
                err100_mean=float(np.mean([r["err100"] for r in rows])), err100_median=float(np.median([r["err100"] for r in rows])),
                off100=float(np.mean([r["off100"] for r in rows])), nonfinite=float(np.mean([not r["finite"] for r in rows])))


class NetD(c2.NetC):
    def __init__(self, model):
        super().__init__(model)
        self.calls = []

    def force(self, sc, p, v, field):
        a, f = super().force(sc, p, v, field)
        an = a.norm(dim=-1)
        w = int(an.argmax())
        r = p.abs().amax(-1)
        self.calls.append(dict(tot=float(an.max()), fld=float(self.store["out"][0, 0].abs().max()),
                               worst_nn=float(eg.nn_dist(p)[w]), worst_r=float(r[w]), n_gt200=int((an > 200).sum()), n_gt200_off=int(((an > 200) & (r >= HALF)).sum()),
                               n_gt200_nn1=int(((an > 200) & (eg.nn_dist(p) < 1.0)).sum()), frac_off=float((r >= HALF).double().mean())))
        return a, f


def gravity_rows(net, scenes, steps):
    rows = []
    for s in scenes:
        net.calls = []
        Pm, _ = eg.roll(net, s, steps=steps)
        c = net.calls[1:]
        ep = (Pm - s["P"][: steps + 1]).norm(dim=-1).mean(1).cpu().numpy()
        post = c[20:]
        w = int(np.argmax([x["tot"] for x in post]))
        truth_a = max(float(eg.exact_acc(s["P"][t], s["m"]).norm(dim=-1).max()) for t in range(0, steps + 1, 5))
        rows.append(dict(tot=[x["tot"] for x in post], grid=[x["tot"] for x in post], loc=[0.0], fld=[x["fld"] for x in c], err20=float(ep[20]), err100=float(ep[100]),
                         err150=float(ep[steps]), off100=offgrid(Pm[-1]), finite=bool(torch.isfinite(Pm).all()), worst=post[w], truth_a_max=truth_a,
                         onset=next((i + 21 for i, x in enumerate(post) if x["tot"] > 200), None)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--softcaps", default="0,100")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--no-gravity", action="store_true")
    args = ap.parse_args()
    dev = eg.DEV
    seeds = [int(x) for x in args.seeds.split(",")]
    caps = [float(x) for x in args.softcaps.split(",")]
    model = cs.build("C", dev, args.ckpt, 0.0)
    net = NetD(model)
    t0 = time.time()
    scenes = {}
    if not args.no_gravity:
        for sd in seeds:
            scenes[sd] = {"expB": eg.make_scenes("expB", 12, sd), "cluster": eg.make_scenes("cluster", 6, sd)}
        print("gravity scenes built", round(time.time() - t0), flush=True)
    evals = {sd: cs.evalset("c_id", sd, args.ticks, dev) for sd in seeds} if not args.no_contact else {}
    res = {"ckpt": args.ckpt, "seeds": seeds, "caps": {}}
    for s in caps:
        model.softcap = s
        r = {"contact": {}, "gravity": {}}
        with torch.no_grad():
            for sd, es in evals.items():
                model.net.geo = None
                rows = contact_rows(model, es, args.ticks, dev)
                r["contact"][str(sd)] = rows
                print(f"s={s} contact seed {sd}", json.dumps(summ(rows)), flush=True)
            for sd, sc in scenes.items():
                model.net.geo = None
                model.momfix = True
                model.geom = None
                for fam, ss_ in sc.items():
                    rows = gravity_rows(net, ss_, 150)
                    r["gravity"][f"{fam}_{sd}"] = rows
                    print(f"s={s} gravity {fam} seed {sd}", json.dumps(summ(rows)), flush=True)
        res["caps"][str(s)] = r
        json.dump(res, open(args.out, "w"))
        print(f"s={s} done t={time.time() - t0:.0f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
