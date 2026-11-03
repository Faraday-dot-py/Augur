"""Per-particle decomposition of the hole-4 force error for one checkpoint (window cutouts as hetero_ext_checks; mode ext = true far field
supplied and scored vs full truth, mode iso = old protocol). Splits d = model - truth into radial (outward from the window centre) and
tangential parts and reports the beta numerator <d.Fc> split the same way, per edge bin, local-density tercile (neighbours within 4, pooled
terciles per scene) and window-N. All bodies have mass 1 in these scenes, so there is no mass tercile.
Usage: PYTHONPATH=. python scripts/hetero_ext_diag.py --ckpt S1.pt [--expe] --tag S1 --out diag_S1.json"""
import argparse
import json

import numpy as np
import torch

from scripts import scatter_scaling_ops as ops
from scripts.hetero_ext_checks import far_of, set_far
from scripts.hetero_tier_checks import BINS, DT, EPS, HALF, REPS, SCENES_A, mforce
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling_eval import SUB, load

R = 4.0


@torch.no_grad()
def collect(model, dev, a):
    rows = {}
    t0 = getattr(model, "t0", None) is not None and model.t0.cfg.ext > 0
    modes = ("iso", "ext") if t0 else ("iso",)
    for sname, (nb, sig, ext, centres) in SCENES_A.items():
        for m in modes:
            rows[(sname, m)] = []
        for seed in [int(x) for x in a.seeds.split(",")]:
            for rep in range(REPS):
                p0, v0 = make_scene(nb, sig, 1.0, ext, seed, rep)
                Pt, Vt = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), 20, DT, SUB, EPS)
                for ci, c in enumerate(centres):
                    cc = torch.tensor(c, device=dev, dtype=torch.float64)
                    model.grid, model.extent, model.h = 128, 2 * HALF, 2 * HALF / 128
                    fields = {m: model.init_field(1, dev)[0] for m in modes}
                    for t in range(21):
                        rel = Pt[t] - cc
                        idx = (rel.abs() < HALF).all(-1).nonzero()[:, 0]
                        pw, vw = rel[idx], Vt[t][idx]
                        t_iso = ops.exact_acc(pw, EPS)
                        t_full = ops.exact_acc(Pt[t], EPS)[idx]
                        F = t_full - t_iso
                        Fc = F - F.mean(0, keepdim=True)
                        edge = HALF - pw.abs().max(-1).values
                        dens = ((torch.cdist(pw, pw) < R).sum(-1) - 1).double()
                        rh = pw / pw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        th = torch.stack([-rh[:, 1], rh[:, 0]], -1)
                        for m in modes:
                            if m == "ext":
                                set_far(model, far_of(Pt[t], cc), 3)
                            elif t0:
                                model.t0.ext_grid = model.t0.ext_poly = model.t0.ext_high = None
                            am, fields[m] = mforce(model, pw.float(), vw.float(), fields[m])
                            if t not in (5, 10, 20):
                                continue
                            d = am.double() - (t_iso if m == "iso" else t_full)
                            if m == "ext":
                                ea = model.t0.ext_acc(model, pw.float()[None])[0].double()
                                de = ea - F
                                dn = d - de
                            else:
                                dn, de = d, torch.zeros_like(d)
                            rows[(sname, m)].append(torch.stack([
                                edge, dens, (d * rh).sum(-1), (d * th).sum(-1), (Fc * rh).sum(-1), (Fc * th).sum(-1),
                                (F * rh).sum(-1), torch.full_like(edge, len(idx)),
                                (dn * rh).sum(-1), (dn * th).sum(-1), (de * rh).sum(-1), (de * th).sum(-1)], -1).cpu().numpy())
        print("done", sname, flush=True)
    return {k: np.concatenate(v) for k, v in rows.items()}


def summarise(rows, tag):
    out = {}
    for (sc, m), A in rows.items():
        edge, dens, dr, dt_, fr, ft, Fr, N, nr, nt, er, et = A.T
        cuts = np.percentile(dens, [33.3, 66.7])
        ter = np.digitize(dens, cuts)
        for b, (lo, hi) in enumerate(BINS):
            for tn, sel_t in (("all", np.ones_like(ter, bool)), ("dens_lo", ter == 0), ("dens_mid", ter == 1), ("dens_hi", ter == 2)):
                s = (edge >= lo) & (edge < hi) & sel_t
                if s.sum() < 20:
                    continue
                den = (fr[s] ** 2 + ft[s] ** 2).sum()
                out[f"{sc}/{m}/bin{lo}-{hi}/{tn}"] = dict(
                    n=int(s.sum()), beta=float((dr[s] * fr[s] + dt_[s] * ft[s]).sum() / den), beta_rad=float((dr[s] * fr[s]).sum() / den),
                    beta_tan=float((dt_[s] * ft[s]).sum() / den), mean_dr=float(dr[s].mean()), mean_dt=float(dt_[s].mean()),
                    mean_Fr=float(Fr[s].mean()), rms_F=float(np.sqrt((fr[s] ** 2 + ft[s] ** 2).mean())), mean_dens=float(dens[s].mean()),
                    cut=[float(c) for c in cuts],
                    beta_net=float((nr[s] * fr[s] + nt[s] * ft[s]).sum() / den), beta_ext_trunc=float((er[s] * fr[s] + et[s] * ft[s]).sum() / den))
    for k, v in out.items():
        if k.endswith("/all") or "bin8-16" in k or "bin16-32" in k:
            print(f"DIAG {tag} {k}: n {v['n']} beta {v['beta']:+.3f} (rad {v['beta_rad']:+.3f} tan {v['beta_tan']:+.3f}) mean d_r {v['mean_dr']:+.4f} d_t {v['mean_dt']:+.4f} mean F_r {v['mean_Fr']:+.4f} net {v['beta_net']:+.3f} trunc {v['beta_ext_trunc']:+.3f} rmsF {v['rms_F']:.4f} dens {v['mean_dens']:.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expe", action="store_true")
    ap.add_argument("--seeds", default="9000,12000,13000,14000,15000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(4738)
    if a.expe:
        model = load(a.ckpt, dev)
    else:
        from scripts import scatter_tier0 as t0m
        from scripts import train_scatter_field as tsf

        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        model = t0m.attach_tier0(tsf.build(tsf.EXPS["E"], "ms_kp_pot_v_g128", DT).to(dev), ck["t0"])
        model.load_state_dict(ck["model"])
        model.eval()
    rows = collect(model, dev, a)
    json.dump(summarise(rows, a.tag), open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
