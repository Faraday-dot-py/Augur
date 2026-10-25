"""Which part of the scatter-field force path carries the solid error (docs/debugging/experiment-log.md, hole-1 follow-ups).
At truth states of each scene (frames 0,100,400) the force of an energy-mode checkpoint is recomputed with one component swapped or removed and
compared to the true LJ force (LJ units, mean removed): full; mesh off; mesh psi-net only; mesh FFT kernel only; pair replaced by exact LJ;
exact pair + mesh off (sanity, ~0); exact pair + net only. Also the full-model error binned by the particle's sub-cell position (CIC corner
distance, 4 bins) and a rollout err@20 with exact LJ force on the model integrator (dt 0.025) as the integrator floor.
Usage: PYTHONPATH=. python -m scripts.scatter_lj_channel_diag --ckpts P=ckpt/P_final.pt,M=ckpt/M_final.pt --tag ch"""
import argparse
import json

import numpy as np
import torch

from scripts import lj_eval, lj_sim
from scripts import scatter_lj as sl


def rms(t):
    return float((t ** 2).sum(-1).mean().sqrt())


def force_variant(m, pt, pos, vel, mask, bm, lay, st, nomesh, oracle, part):
    m.nomesh, m.oracle, m.mesh_part = nomesh, oracle, part
    with torch.no_grad():
        a, _ = pt.force(pos, vel, mask, bm, lay, st)
    pt.take_pen()
    return a[0] * sl.LS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenes", default="solid,liquid,melt")
    ap.add_argument("--frames", default="0,100,400")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = {}
    variants = {"full": (False, False, "all"), "mesh_off": (True, False, "all"), "net_only": (False, False, "net"), "kernel_only": (False, False, "kernel"),
                "exact_pair": (False, True, "all"), "exact_pair_mesh_off": (True, True, "all"), "exact_pair_net_only": (False, True, "net")}
    for kv in args.ckpts.split(","):
        label, path = kv.split("=")
        m = sl.load_model(path, dev)
        m.eval()
        if not getattr(m, "energy", False):
            continue
        pt = sl.PeriodicTiles(m)
        res = {}
        for name in args.scenes.split(","):
            z = np.load(f"results/lj_{name}.npz")
            box = torch.tensor(z["box"], device=dev, dtype=torch.float64)
            lay = sl.layout((box / sl.LS).cpu().numpy())
            bm = (box / sl.LS)[None]
            for fr in [int(f) for f in args.frames.split(",")]:
                fr = min(fr, len(z["pos"]) - 1)
                p = torch.tensor(z["pos"][fr], device=dev, dtype=torch.float64)
                v = torch.tensor(z["vel"][fr], device=dev, dtype=torch.float64)
                at, _, _ = lj_sim.force(p, box)
                at = at - at.mean(0)
                pm, vm = (p / sl.LS)[None], (v / sl.LS)[None]
                mask = torch.ones(1, len(p), device=dev, dtype=torch.float64)
                st = pt.init_field(1, lay, dev)
                r = {"rms_true": rms(at)}
                for vn, (nm, orc, part) in variants.items():
                    r[vn] = rms(force_variant(m, pt, pm, vm, mask, bm, lay, st, nm, orc, part) - at)
                m.nomesh, m.oracle, m.mesh_part = False, False, "all"
                e = force_variant(m, pt, pm, vm, mask, bm, lay, st, False, False, "all") - at
                fx = (pm[0, :, 0] / m.h) % 1.0
                fy = (pm[0, :, 1] / m.h) % 1.0
                dist = torch.sqrt((fx - 0.5) ** 2 + (fy - 0.5) ** 2)
                qs = torch.quantile(dist, torch.tensor([0.25, 0.5, 0.75], device=dev, dtype=dist.dtype))
                bins = torch.bucketize(dist, qs)
                r["err_by_subcell_quartile_center_to_corner"] = [rms(e[bins == b]) for b in range(4)]
                res[f"{name}:{fr}"] = r
                print(label, name, fr, {k: (round(x, 4) if isinstance(x, float) else [round(y, 4) for y in x]) for k, x in r.items()}, flush=True)
        out[label] = res
    z = np.load("results/lj_solid.npz")
    mi = sl.build_model(energy=True, oracle=True, nomesh=True)
    mi = mi.to(dev)
    box = z["box"]
    nf = 20
    fp, _, _, ok = sl.rollout(mi, z["pos"][0].astype(np.float64), z["vel"][0].astype(np.float64), box, nf, dev)
    out["integrator_floor_solid_err20"] = float(lj_eval.err_curve(fp[nf], z["pos"][nf], box).mean())
    print("integrator floor (exact LJ, dt 0.025) solid err@20", out["integrator_floor_solid_err20"], flush=True)
    json.dump(out, open(f"results/scatter_lj_channel_diag_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
