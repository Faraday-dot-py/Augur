"""Phase 1 CFD contact-generalization training: ball-ball contact (varying
mass/radius), ball-vs-circle-obstacle, and ball-vs-wall-segment-obstacle
scenes, truth from scripts/contact_truth.py (bounce.py's own physics, GPU
vectorized), model from model/contact_force.py. Boundary-point density is
swept per scene (spec §4/§6/§9 resolved decision), not fixed.

Usage: PYTHONPATH=. python3 -m scripts.train_contact_dynamics --device cuda
"""
import argparse
import json
import math
import random

import torch

from model import obstacles as ob
from model.contact_force import ContactForceDynamics, ContactForceDynamicsSymlog, ContactForceDynamicsSymlogCapped
from scripts import contact_truth as ct

MASS_RANGE = (0.25, 4.0)  # train range per docs/debugging/z-and-mass-channels-feasibility.md recommendation
RADIUS_RANGE = (0.4, 1.2)
SPACING_RANGE = (0.6, 2.0)  # swept boundary-point spacing (spec §4 boundary-sampling question)
CENTER = 500.0  # scene offset away from bounce.py's lo=0 walls, as in scripts/gravity_sim.py
SPAWN_TRIES = 20


def make_scene(rng, min_bodies, max_bodies, n, obstacle_prob=2.0 / 3.0, circle_radius_range=(2.0, 6.0),
               spacing_range=SPACING_RANGE):
    """One training scene: a handful of real balls, plus (with some
    probability) one obstacle -- a circle or a finite wall segment,
    injected as extra kinematic point-cloud entries in the same ball list
    scripts/contact_truth.py already understands. The scene sits at
    [CENTER, CENTER+n], far from bounce.py's lo=0 walls. The obstacle is
    placed first so real balls can be re-drawn (up to SPAWN_TRIES times)
    until they overlap nothing already placed. obstacle_prob/circle_radius_range/
    spacing_range default to the original v1/v2/symlog distribution (1/3 circle,
    1/3 wall, 1/3 none; radius 2.0-6.0; spacing SPACING_RANGE) -- widened by the
    densershape variant to give the symlog head more obstacle-shape coverage,
    per docs/debugging/contact-force-architecture-ideas-untested.md."""
    obstacle = []
    kind = "none" if rng.random() >= obstacle_prob else rng.choice(["circle", "wall"])
    spacing = rng.uniform(*spacing_range)
    point_radius = ob.default_point_radius(spacing)
    if kind == "circle":
        cx, cy, cr = rng.uniform(0.2 * n, 0.8 * n), rng.uniform(0.2 * n, 0.8 * n), rng.uniform(*circle_radius_range)
        pts, radii = ob.sample_circle_boundary(CENTER + cx, CENTER + cy, cr, spacing, point_radius)
        for (x, y), r in zip(pts, radii):
            obstacle.append({"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "radius": float(r), "kinematic": True})
    elif kind == "wall":
        orientation = rng.choice(["h", "v"])
        coord = rng.uniform(0.3 * n, 0.7 * n)
        lo = rng.uniform(0.1 * n, 0.4 * n)
        hi = lo + rng.uniform(0.2 * n, 0.4 * n)
        pts, radii = ob.sample_wall_segment_boundary(orientation, CENTER + coord, CENTER + lo, CENTER + hi,
                                                     spacing, point_radius)
        for (x, y), r in zip(pts, radii):
            obstacle.append({"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "radius": float(r), "kinematic": True})
    num_real = rng.randint(min_bodies, max_bodies)
    balls = []
    for _ in range(num_real):
        r = rng.uniform(*RADIUS_RANGE)
        m = math.exp(rng.uniform(math.log(MASS_RANGE[0]), math.log(MASS_RANGE[1])))
        for _ in range(SPAWN_TRIES):
            x = CENTER + rng.uniform(2 * r, n - 1 - 2 * r)
            y = CENTER + rng.uniform(2 * r, n - 1 - 2 * r)
            if all(math.hypot(x - b["x"], y - b["y"]) >= r + b["radius"] for b in balls + obstacle):
                break
        balls.append({
            "x": x, "y": y,
            "vx": rng.uniform(-3.0, 3.0), "vy": rng.uniform(-3.0, 3.0),
            "radius": r, "mass": m,
        })
    return balls + obstacle


def build_truth(balls, steps, dt, gravity, stiffness, substeps, device):
    pos, vel = ct.rollout(balls, n=2000, steps=steps, dt=dt, gravity=gravity, stiffness=stiffness,
                           substeps=substeps, segments=None, device=device)
    pos = torch.tensor(pos, dtype=torch.float32, device=device)
    vel = torch.tensor(vel, dtype=torch.float32, device=device)
    radius = torch.tensor([b.get("radius", 0.75) for b in balls], dtype=torch.float32, device=device)
    mass = torch.tensor([b.get("mass", 1.0) for b in balls], dtype=torch.float32, device=device)
    kinematic = torch.tensor([b.get("kinematic", False) for b in balls], dtype=torch.bool, device=device)
    return pos, vel, radius, mass, kinematic


def unroll(dyn, pos, vel, radius, mass, kinematic, k, dt):
    hidden = torch.zeros(pos.shape[0], dyn.hidden_dim, device=pos.device)
    ps, vs = [], []
    for _ in range(k):
        dp, dv, hidden = dyn(pos, vel, hidden, radius, mass, kinematic)
        pos = pos + vel * dt + dp
        vel = vel + dv
        ps.append(pos)
        vs.append(vel)
    return torch.stack(ps), torch.stack(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bodies", type=int, default=2)
    ap.add_argument("--max-bodies", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gravity", type=float, default=0.0)
    ap.add_argument("--stiffness", type=float, default=400.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--neighbor-radius", type=float, default=6.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--checkpoint", default="checkpoints/contact_dynamics_v1.pt")
    ap.add_argument("--out", default="results/contact_dynamics_v1.json")
    ap.add_argument("--model", default="linear", choices=["linear", "symlog", "symlog_capped"])
    ap.add_argument("--log-scale-cap", type=float, default=4.0)  # symlog_capped only
    ap.add_argument("--obstacle-prob", type=float, default=2.0 / 3.0)  # denser-obstacle-shape variant
    ap.add_argument("--circle-radius-min", type=float, default=2.0)
    ap.add_argument("--circle-radius-max", type=float, default=6.0)
    ap.add_argument("--spacing-min", type=float, default=SPACING_RANGE[0])
    ap.add_argument("--spacing-max", type=float, default=SPACING_RANGE[1])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model_cls = {"linear": ContactForceDynamics, "symlog": ContactForceDynamicsSymlog,
                 "symlog_capped": ContactForceDynamicsSymlogCapped}[args.model]
    model_kwargs = {"log_scale_cap": args.log_scale_cap} if args.model == "symlog_capped" else {}
    dyn = model_cls(dt=args.dt, neighbor_radius=args.neighbor_radius, **model_kwargs).to(args.device)
    opt = torch.optim.Adam(dyn.parameters(), lr=args.lr)
    loss_history = []
    for it in range(args.iters):
        k = int(round(args.k_start + (args.k_end - args.k_start) * it / max(1, args.iters - 1)))
        opt.zero_grad()
        total = 0.0
        skipped = 0
        for _ in range(args.batch):
            balls = make_scene(rng, args.min_bodies, args.max_bodies, n=40, obstacle_prob=args.obstacle_prob,
                               circle_radius_range=(args.circle_radius_min, args.circle_radius_max),
                               spacing_range=(args.spacing_min, args.spacing_max))
            pos, vel, radius, mass, kinematic = build_truth(balls, args.steps, args.dt, args.gravity,
                                                              args.stiffness, args.substeps, args.device)
            t0 = rng.randint(0, args.steps - k)
            ps, vs = unroll(dyn, pos[t0].clone(), vel[t0].clone(), radius, mass, kinematic, k, args.dt)
            tp = pos[t0 + 1:t0 + k + 1].clone()
            tv = vel[t0 + 1:t0 + k + 1].clone()
            loss = (((ps - tp) ** 2).mean() + 0.1 * ((vs - tv) ** 2).mean()) / args.batch
            if not loss.requires_grad:
                skipped += 1
                continue  # scene had no contact pairs within neighbor_radius for the whole horizon
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        if it % args.log_every == 0:
            print(f"it {it} k {k} loss {total:.6f} skipped {skipped}/{args.batch}", flush=True)
            loss_history.append([it, total])
            torch.save(dyn.state_dict(), args.checkpoint)
    torch.save(dyn.state_dict(), args.checkpoint)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "loss_history": loss_history}, f)


if __name__ == "__main__":
    main()
