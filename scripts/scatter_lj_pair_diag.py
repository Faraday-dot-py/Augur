"""Which pair distances carry the scatter-field force error on LJ (docs/debugging/scatter-lj.md, hole-1 section). For truth states of each scene, the pair-term
curve error d(r) = f_model_pair(r) - f_true(r) (curves from the eval json "force", LJ units) is applied to every min-image neighbour pair within rc; per body the
vector sum is binned by pair distance. Reports RMS per-body contribution per r-bin, the RMS true force, the curve-error total, and the full model force error
(mesh + pair + mean removal) at the same states.
Usage: PYTHONPATH=. python -m scripts.scatter_lj_pair_diag --ckpt ckpt/P_final.pt --eval-json results/scatter_lj_h1_P.json --tag P"""
import argparse
import json

import numpy as np
import torch

from scripts import lj_sim
from scripts import scatter_lj as sl

BINS = [0.0, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.8, 2.2, 2.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenes", default="solid,liquid,melt,coexist")
    ap.add_argument("--frames", default="0,100,400")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    fj = json.load(open(args.eval_json))["force"]
    dgrid = np.array(fj["d"])
    curve = np.array(fj["model_pair_term"]) - np.array(fj["true"])
    model = sl.load_model(args.ckpt, dev)
    model.eval()
    out = {}
    for name in args.scenes.split(","):
        z = np.load(f"results/lj_{name}.npz")
        box = torch.tensor(z["box"], device=dev, dtype=torch.float64)
        res = {}
        for fr in [int(f) for f in args.frames.split(",")]:
            fr = min(fr, len(z["pos"]) - 1)
            p = torch.tensor(z["pos"][fr], device=dev, dtype=torch.float64)
            v = torch.tensor(z["vel"][fr], device=dev, dtype=torch.float64)
            n = len(p)
            at, _, _ = lj_sim.force(p, box)
            at = at - at.mean(0)
            dd = p[:, None, :] - p[None, :, :]
            dd = dd - box * torch.round(dd / box)
            r = dd.norm(dim=-1)
            r.fill_diagonal_(1e9)
            err_r = torch.tensor(np.interp(r.clamp(max=2.5).cpu().numpy(), dgrid, curve), device=dev, dtype=torch.float64) * (r < 2.5)
            unit = dd / r[..., None]
            bins = {}
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                sel = (r >= lo) & (r < hi)
                c = (err_r * sel)[..., None] * unit
                c = c.sum(1)
                bins[f"{lo}-{hi}"] = {"rms_contrib": float((c ** 2).sum(-1).mean().sqrt()), "pairs_per_body": float(sel.sum(1).float().mean())}
            tot = (err_r[..., None] * unit).sum(1)
            lay = sl.layout((box / sl.LS).cpu().numpy())
            pt = sl.PeriodicTiles(model)
            pm, vm = (p / sl.LS)[None], (v / sl.LS)[None]
            mask = torch.ones(1, n, device=dev, dtype=torch.float64)
            st = (pt.init_field(1, lay, dev), None)
            with torch.no_grad():
                for _ in range(8):
                    pm, vm, st = pt.step(pm, vm, mask, (box / sl.LS)[None], lay, st)
            at2, _, _ = lj_sim.force(pm[0] * sl.LS, box)
            at2 = at2 - at2.mean(0)
            am, _ = pt.force(pm, vm, mask, (box / sl.LS)[None], lay, st[0])
            am = am[0] * sl.LS
            rms = lambda t: float((t ** 2).sum(-1).mean().sqrt())
            res[fr] = {"rms_true_force": rms(at), "rms_curve_error_total": rms(tot), "rms_full_model_error_at_stepped_state": rms(am - at2),
                       "rms_true_force_stepped": rms(at2), "bins": bins}
            print(name, fr, {k: v for k, v in res[fr].items() if k != "bins"}, flush=True)
            print("  ", {k: round(b["rms_contrib"], 4) for k, b in bins.items()}, flush=True)
        out[name] = res
    json.dump(out, open(f"results/scatter_lj_pair_diag_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
