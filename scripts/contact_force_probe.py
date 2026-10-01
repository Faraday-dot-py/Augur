"""Direct contact-force probe at the contact point (pen=0, d=r_sum): mass1
vs mass2, report the signed pairwise force along the pair axis (from
-dV/dd, see model/contact_force.py ContactForceDynamics.accel). This is
the same diagnostic used to root-cause the seed-12000 mass-ratio OOD blind
spot (docs/debugging/contact-force-architecture-ideas-untested.md,
project_cfd_contact_gen_variant_status.md memory): v2 head +9.65 (correct,
repulsive), symlog (uncapped, pre-densershape) +1.31, symlog_densershape
-0.09 (near-zero, wrong sign -- lets bodies interpenetrate).

Reports the raw net force on the mass1 body (pre mass-divide, i.e. before
accel()'s final `/ mass`), matching the earlier-reported numbers' convention.
Positive = repulsive (pushes bodies apart).

Usage: PYTHONPATH=. python3 -m scripts.contact_force_probe \
    --checkpoint checkpoints/contact_dynamics_symlog_densershape.pt --model symlog \
    --mass1 1.0 --mass2 10.0
"""
import argparse

import torch

from model.contact_force import ContactForceDynamics, ContactForceDynamicsSymlog, ContactForceDynamicsSymlogCapped


def probe(dyn, mass1, mass2, radius1=0.75, radius2=0.75, device="cpu"):
    r_sum = radius1 + radius2
    positions = torch.tensor([[0.0, 0.0], [r_sum, 0.0]], device=device)  # d = r_sum, pen = 0
    radius = torch.tensor([radius1, radius2], device=device)
    mass = torch.tensor([mass1, mass2], device=device)
    edges = torch.tensor([[0, 1], [1, 0]], device=device)  # both directed edges, like build_radius_graph_cells

    rel = positions[edges[0]] - positions[edges[1]]
    d = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
    d.requires_grad_(True)
    rs = (radius[edges[0]] + radius[edges[1]]).unsqueeze(-1)
    pen = rs - d
    feat = dyn.pair_features(pen, rs, mass[edges[0]].unsqueeze(-1), mass[edges[1]].unsqueeze(-1))
    gate = torch.sigmoid(pen / 0.5)
    v = dyn.potential(feat) * gate
    (grad_d,) = torch.autograd.grad(v.sum(), d)
    force = grad_d * rel / d
    net = torch.zeros_like(positions).index_add(0, edges[1], force)
    return net[0, 0].item()  # x-component of net force on body 0 (mass1), body 1 sits at +x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="symlog", choices=["linear", "symlog", "symlog_capped"])
    ap.add_argument("--log-scale-cap", type=float, default=4.0)
    ap.add_argument("--mass1", type=float, default=1.0)
    ap.add_argument("--mass2", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_cls = {"linear": ContactForceDynamics, "symlog": ContactForceDynamicsSymlog,
                 "symlog_capped": ContactForceDynamicsSymlogCapped}[args.model]
    model_kwargs = {"log_scale_cap": args.log_scale_cap} if args.model == "symlog_capped" else {}
    dyn = model_cls(**model_kwargs)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    dyn.to(args.device)

    force = probe(dyn, args.mass1, args.mass2, args.radius, args.radius, args.device)
    print(f"mass1={args.mass1} mass2={args.mass2} d=r_sum={2 * args.radius} force={force:.4f}")


if __name__ == "__main__":
    main()
