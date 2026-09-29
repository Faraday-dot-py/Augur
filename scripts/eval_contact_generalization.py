# scripts/eval_contact_generalization.py
"""Phase 1 held-out generalization eval: unseen obstacle shapes/positions/
boundary-point densities, unseen mass/radius ratios, seeds 9000/12000
(per [[feedback_select_on_heldout_seeds]]), plus an O(N)/tiling sanity
check that boundary point count doesn't blow up build_radius_graph_cells'
per-cell assumptions (spec §7).

Usage: PYTHONPATH=. python3 -m scripts.eval_contact_generalization --checkpoint checkpoints/contact_dynamics_v1.pt --device cuda
"""
import argparse
import json
import random

import numpy as np
import torch

from model import obstacles as ob
from model.contact_force import ContactForceDynamics, ContactForceDynamicsSymlog, ContactForceDynamicsSymlogCapped
from model.token_graph import build_radius_graph_cells
from scripts import contact_truth as ct
from scripts.train_contact_dynamics import unroll


def scenario_unseen_obstacle_shape(rng, n=40):
    """Held-out: a much bigger circle at a swept density outside training's
    SPACING_RANGE (0.6-2.0), and a ball aimed straight at it -- the actual
    target capability (spec §0/§7): drop a shape in after training. Spacing
    is jittered per seed (2.5-4.0, still above training's 2.0 ceiling) so
    seeds 9000/12000 aren't byte-identical scenes -- a review finding
    caught this when both seeds produced identical numbers, defeating the
    point of a held-out-seed sweep (obstacle radius/position/ball path stay
    fixed at their verified-to-contact-within-horizon values)."""
    spacing = rng.uniform(2.5, 4.0)
    point_radius = ob.default_point_radius(spacing)
    pts, radii = ob.sample_circle_boundary(20.0, 20.0, 10.0, spacing, point_radius)
    balls = [{"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "radius": float(r), "kinematic": True}
             for (x, y), r in zip(pts, radii)]
    balls.append({"x": 5.0, "y": 20.0, "vx": 4.0, "vy": 0.0, "radius": 0.75, "mass": 1.0})
    return balls


def scenario_unseen_mass_ratio(rng, n=40):
    """Held-out: mass ratio 1:10 or 10:1 (train range 0.25-4), chosen per
    seed so 9000/12000 differ. Initial gap (2.5, closing at vx=2.0 -> contact
    around step 10) was shortened from an earlier draft that put the pair
    5.0 apart -- at the default --steps 20/dt 0.05, that gap needed 35 steps
    to close and the two bodies never actually touched within the eval
    horizon, silently testing free-flight instead of mass-ratio-dependent
    contact response. A review finding caught this by directly checking the
    minimum inter-body distance over the rollout."""
    ratio = rng.choice([10.0, 0.1])
    return [
        {"x": 17.5, "y": 15.0, "vx": 2.0, "vy": 0.0, "radius": 0.75, "mass": 1.0},
        {"x": 20.0, "y": 15.3, "vx": 0.0, "vy": 0.0, "radius": 0.75, "mass": 1.0 / ratio},
    ]


def scenario_unseen_wall_density(rng, n=40):
    """Held-out: wall segment sampled at a jittered density (3.0-5.0),
    above training's 0.6-2.0 range -- tests robustness to sparser-than-
    trained sampling (spec §4 boundary-sampling question #3). Spacing is
    jittered per seed (approach speed/positions stay fixed: the original
    vy=12.0 barely closes the 15-unit gap within the eval horizon, so a
    wider jitter there risked reintroducing the no-contact bug found in
    scenario_unseen_mass_ratio -- see main()'s --steps default, raised to
    30 for margin)."""
    spacing = rng.uniform(3.0, 5.0)
    point_radius = ob.default_point_radius(spacing)
    pts, radii = ob.sample_wall_segment_boundary("h", 20.0, 5.0, 35.0, spacing, point_radius)
    balls = [{"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "radius": float(r), "kinematic": True}
             for (x, y), r in zip(pts, radii)]
    balls.append({"x": 20.0, "y": 5.0, "vx": 0.0, "vy": 12.0, "radius": 0.75, "mass": 1.0})
    return balls


SCENARIOS = {
    "unseen_obstacle_shape": scenario_unseen_obstacle_shape,
    "unseen_mass_ratio": scenario_unseen_mass_ratio,
    "unseen_wall_density": scenario_unseen_wall_density,
}


def evaluate_scenario(dyn, balls, steps, dt, gravity, stiffness, substeps, device):
    truth_pos, truth_vel = ct.rollout(balls, n=1000, steps=steps, dt=dt, gravity=gravity, stiffness=stiffness,
                                       substeps=substeps, segments=None, device=device)
    radius = torch.tensor([b.get("radius", 0.75) for b in balls], dtype=torch.float32, device=device)
    mass = torch.tensor([b.get("mass", 1.0) for b in balls], dtype=torch.float32, device=device)
    kinematic = torch.tensor([b.get("kinematic", False) for b in balls], dtype=torch.bool, device=device)
    pos0 = torch.tensor(truth_pos[0], dtype=torch.float32, device=device)
    vel0 = torch.tensor(truth_vel[0], dtype=torch.float32, device=device)
    with torch.no_grad():
        ps, vs = unroll(dyn, pos0, vel0, radius, mass, kinematic, steps, dt)
    ps, vs = ps.cpu().numpy(), vs.cpu().numpy()
    real = ~kinematic.cpu().numpy()
    err = {t: float(np.linalg.norm(ps[t - 1][real] - truth_pos[t][real], axis=1).mean())
           for t in (5, 10, 20, 30) if t <= steps}
    kin_drift = float(np.abs(ps[:, ~real] - truth_pos[0][None, ~real]).max()) if (~real).any() else 0.0
    return {"err": err, "kinematic_drift": kin_drift}


def check_tiling_scaling(dyn, n_points=(50, 500, 5000)):
    """O(N) sanity check: edge count per node from build_radius_graph_cells
    should stay roughly constant as N grows at fixed density, not blow up
    quadratically once obstacle point clouds add many nodes (spec §7)."""
    out = {}
    for n in n_points:
        pos = torch.rand(n, 2) * (n ** 0.5) * dyn.neighbor_radius
        edges = build_radius_graph_cells(pos, dyn.neighbor_radius)
        out[n] = edges.shape[1] / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/contact_dynamics_v1.pt")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gravity", type=float, default=0.0)  # see Task 5: ContactForceDynamics has no external-field term
    ap.add_argument("--stiffness", type=float, default=400.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--neighbor-radius", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=30)  # raised from 20 for contact-timing margin across scenarios, see scenario_unseen_wall_density
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/contact_generalization_v1.json")
    ap.add_argument("--model", default="linear", choices=["linear", "symlog", "symlog_capped"])
    ap.add_argument("--log-scale-cap", type=float, default=4.0)  # symlog_capped only, must match training
    args = ap.parse_args()

    model_cls = {"linear": ContactForceDynamics, "symlog": ContactForceDynamicsSymlog,
                 "symlog_capped": ContactForceDynamicsSymlogCapped}[args.model]
    model_kwargs = {"log_scale_cap": args.log_scale_cap} if args.model == "symlog_capped" else {}
    dyn = model_cls(dt=args.dt, neighbor_radius=args.neighbor_radius, **model_kwargs)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    dyn.to(args.device)

    results = {}
    for seed in (9000, 12000):
        rng = random.Random(seed)
        for name, builder in SCENARIOS.items():
            balls = builder(rng)
            res = evaluate_scenario(dyn, balls, args.steps, args.dt, args.gravity, args.stiffness,
                                     args.substeps, args.device)
            results[f"{name}_{seed}"] = res
            print(name, seed, res, flush=True)
    results["tiling_scaling"] = check_tiling_scaling(dyn)
    print("edges/node vs N (should stay ~flat):", results["tiling_scaling"], flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
