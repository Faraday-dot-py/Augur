"""Fit per-cell polynomial coefficients to a smooth conservative target force (sum of Gaussian wells), three blending forms, report held-out force error and
loop work. Usage: PYTHONPATH=. python -m scripts.poly_cell_fit --out results/poly_cell_fit.json"""
import argparse
import json

import torch

from scripts import poly_cell_potential as pc

G, H, E = 16, 1.0, 16.0


def target_force(pos, centres, sig, amp):
    d = pos[:, :, None, :] - centres[None, None]
    w = amp * torch.exp(-(d ** 2).sum(-1) / (2 * sig ** 2))
    return (w[..., None] * d / sig ** 2).sum(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/poly_cell_fit.json")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    g = torch.Generator().manual_seed(args.seed)
    res = {}
    for sig in (3.0, 1.5, 0.8):
        centres = ((torch.rand(6, 2, generator=g, dtype=torch.float64) - 0.5) * (E - 4)).to(dev)
        amp = torch.randn(6, generator=g, dtype=torch.float64).to(dev)
        tr = ((torch.rand(1, 20000, 2, generator=g, dtype=torch.float64) - 0.5) * (E - 2)).to(dev)
        te = ((torch.rand(1, 20000, 2, generator=g, dtype=torch.float64) - 0.5) * (E - 2)).to(dev)
        ftr, fte = target_force(tr, centres, sig, amp), target_force(te, centres, sig, amp)
        out = {}
        for name in ("grad_order2", "grad_order1", "naive_blend_order2"):
            coef = torch.zeros(1, 6, G, G, dtype=torch.float64, device=dev, requires_grad=True)
            opt = torch.optim.Adam([coef], lr=0.02)
            for s in range(args.steps):
                c = coef if name != "grad_order1" else torch.cat([coef[:, :3], torch.zeros_like(coef[:, 3:])], 1)
                f = pc.naive_force(c, tr, H, E) if name.startswith("naive") else pc.force(c, tr, H, E, create_graph=True)
                loss = ((f - ftr) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            c = coef.detach() if name != "grad_order1" else torch.cat([coef.detach()[:, :3], torch.zeros_like(coef.detach()[:, 3:])], 1)
            fn = (lambda p, c=c: pc.naive_force(c, p, H, E)) if name.startswith("naive") else (lambda p, c=c: pc.force(c, p, H, E))
            f = fn(te)
            rel = float(((f - fte) ** 2).sum().sqrt() / (fte ** 2).sum().sqrt())
            ctr = torch.tensor([0.5, -0.5], dtype=torch.float64, device=dev)
            lw = [pc.loop_work(fn, ctr, r, H, E) for r in (1.5, 3.0)]
            mag = float((f ** 2).sum(-1).sqrt().mean())
            out[name] = {"heldout_rel_err": rel, "loop_work_r1.5": lw[0], "loop_work_r3": lw[1], "mean_force_mag": mag}
            print(sig, name, out[name], flush=True)
        res[f"sigma_{sig}"] = out
    json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
