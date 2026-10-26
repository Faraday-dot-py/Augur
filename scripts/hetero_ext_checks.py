"""Hole-4 re-score under the imposed-field protocol for tier-0 models trained with an imposed far field (windows / ext arms).
Window cutouts as in hetero_tier_checks phase a (HALF 32, ticks 5/10/20), but the model is fed the true far-field Taylor expansion about the
window centre (direct sum over all matter outside the window, third order unless --order 2) and scored against the FULL-scene truth.
Modes: iso (no ext, vs isolated truth: the old protocol), ext (true far field, vs full truth), rand (far field of another window of the same
scene, vs full truth). beta = <d.Fc>/<Fc.Fc> (as the old protocol: F = full - isolated truth, mean over the window removed; 'raw' keeps the mean), d = model - truth. Exp E (no t0) only gets iso.
Usage: PYTHONPATH=. python scripts/hetero_ext_checks.py --ckpt S1.pt --tag S1 --seeds 9000,12000,13000,14000,15000 --out hext_S1.json [--order 2]
       PYTHONPATH=. python scripts/hetero_ext_checks.py --ckpt E.pt --expe --tag ExpE ..."""
import argparse
import json

import torch

from scripts import scatter_scaling_ops as ops
from scripts.hetero_tier_checks import BINS, EPS, HALF, SCENES_A, DT, REPS, mforce
from scripts.scatter_scaling import make_scene
from scripts.scatter_tiling_eval import SUB, load
from scripts.tier0_harvest import far_third


def far_of(Pt, cc):
    rel = Pt - cc
    out = (rel.abs() >= HALF).any(-1)
    z = torch.zeros(1, 2, device=Pt.device, dtype=Pt.dtype)
    return far_third(z, rel[None], torch.ones(1, rel.shape[0], device=Pt.device, dtype=Pt.dtype), out[None].to(Pt.dtype), EPS)


def set_far(model, far, order):
    a0, J, Q, C = [x.float() for x in far]
    H = torch.stack([-J[:, 0, 0], -0.5 * (J[:, 0, 1] + J[:, 1, 0]), -J[:, 1, 1]], -1)
    model.t0.set_external(model, phi0=torch.zeros(1, device=a0.device), g=-a0, H=H, high=(Q, C) if order == 3 else None)


@torch.no_grad()
def run(model, dev, a):
    rows = []
    t0 = getattr(model, "t0", None) is not None and model.t0.cfg.ext > 0
    modes = ("iso", "ext", "rand") if t0 else ("iso",)
    for sname, (nb, sig, ext, centres) in SCENES_A.items():
        if sname not in a.scenes.split(","):
            continue
        for seed in [int(x) for x in a.seeds.split(",")]:
            for rep in range(REPS):
                p0, v0 = make_scene(nb, sig, 1.0, ext, seed, rep)
                Pt, Vt = ops.exact_rollout(torch.tensor(p0, device=dev), torch.tensor(v0, device=dev), 20, DT, SUB, EPS)
                for ci, c in enumerate(centres):
                    cc = torch.tensor(c, device=dev, dtype=torch.float64)
                    cr = torch.tensor(centres[(ci + 1) % len(centres)], device=dev, dtype=torch.float64)
                    ops.retarget(model, 128, 2 * HALF)
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
                        for m in modes:
                            if m == "ext":
                                set_far(model, far_of(Pt[t], cc), a.order)
                            elif m == "rand":
                                set_far(model, far_of(Pt[t], cr), a.order)
                            elif t0:
                                model.t0.ext_grid = model.t0.ext_poly = model.t0.ext_high = None
                            am, fields[m] = mforce(model, pw.float(), vw.float(), fields[m])
                            if t not in (5, 10, 20):
                                continue
                            tr = t_iso if m == "iso" else t_full
                            d = am.double() - tr
                            for bi, (lo, hi) in enumerate(BINS):
                                mk = (edge >= lo) & (edge < hi)
                                if not mk.any():
                                    continue
                                rows.append({"scene": sname, "seed": seed, "rep": rep, "centre": ci, "tick": t, "bin": bi, "mode": m,
                                             "n": int(mk.sum()), "num": float((d[mk] * F[mk]).sum()), "den": float((F[mk] ** 2).sum()),
                                             "numc": float((d[mk] * Fc[mk]).sum()), "denc": float((Fc[mk] ** 2).sum()),
                                             "e": float(d[mk].norm(dim=-1).sum()), "a": float(tr[mk].norm(dim=-1).sum()),
                                             "F": float(F[mk].norm(dim=-1).sum())})
                print("done", sname, seed, rep, flush=True)
    return rows


def table(rows, tag):
    for sc in ("sparse", "mid", "dense"):
        for m in ("iso", "ext", "rand"):
            g = [r for r in rows if r["scene"] == sc and r["mode"] == m]
            if not g:
                continue
            out = []
            for b in range(4):
                gb = [r for r in g if r["bin"] == b]
                bc = sum(r["numc"] for r in gb) / max(sum(r["denc"] for r in gb), 1e-12)
                br = sum(r["num"] for r in gb) / max(sum(r["den"] for r in gb), 1e-12)
                rel = sum(r["e"] for r in gb) / max(sum(r["a"] for r in gb), 1e-12)
                out.append(f"{BINS[b][0]}-{BINS[b][1]} {bc:+.3f} raw {br:+.3f} err {rel:.3f}")
            print(f"HEXT {tag} {sc} {m}: " + " | ".join(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expe", action="store_true")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--seeds", default="9000,12000,13000,14000,15000")
    ap.add_argument("--scenes", default="sparse,mid,dense")
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
    rows = run(model, dev, a)
    json.dump({"tag": a.tag, "order": a.order, "rows": rows}, open(a.out, "w"))
    table(rows, a.tag)
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
