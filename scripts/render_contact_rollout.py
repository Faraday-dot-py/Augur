# scripts/render_contact_rollout.py
"""Side-by-side truth-vs-model video for one Phase 1 contact-gen scene (a
real ball bouncing off a dropped-in circle obstacle, unseen at training
density per scripts/eval_contact_generalization.py's
scenario_unseen_obstacle_shape). Streams one frame at a time to the video
writer instead of stacking frames in RAM (per [[feedback_local_render_memory_bounds]]).

Usage: PYTHONPATH=. python3 -m scripts.render_contact_rollout --checkpoint checkpoints/contact_dynamics_v1.pt --out videos/contact_dynamics_v1_rollout.mp4
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FFMpegWriter

from model.contact_force import ContactForceDynamics
from scripts import contact_truth as ct
from scripts.eval_contact_generalization import SCENARIOS
from scripts.train_contact_dynamics import unroll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/contact_dynamics_v1.pt")
    ap.add_argument("--scenario", default="unseen_obstacle_shape", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gravity", type=float, default=0.0)  # see Task 5: ContactForceDynamics has no external-field term
    ap.add_argument("--stiffness", type=float, default=400.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--neighbor-radius", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="videos/contact_dynamics_v1_rollout.mp4")
    args = ap.parse_args()

    import random
    balls = SCENARIOS[args.scenario](random.Random(args.seed))
    truth_pos, _ = ct.rollout(balls, n=1000, steps=args.steps, dt=args.dt, gravity=args.gravity,
                               stiffness=args.stiffness, substeps=args.substeps, segments=None, device=args.device)
    radius = torch.tensor([b.get("radius", 0.75) for b in balls], dtype=torch.float32, device=args.device)
    mass = torch.tensor([b.get("mass", 1.0) for b in balls], dtype=torch.float32, device=args.device)
    kinematic = torch.tensor([b.get("kinematic", False) for b in balls], dtype=torch.bool, device=args.device)

    dyn = ContactForceDynamics(dt=args.dt, neighbor_radius=args.neighbor_radius)
    dyn.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    dyn.to(args.device)
    pos0 = torch.tensor(truth_pos[0], dtype=torch.float32, device=args.device)
    vel0 = torch.tensor([[b.get("vx", 0.0), b.get("vy", 0.0)] for b in balls], dtype=torch.float32, device=args.device)
    with torch.no_grad():
        model_pos, _ = unroll(dyn, pos0, vel0, radius, mass, kinematic, args.steps, args.dt)
    model_pos = model_pos.cpu().numpy()

    real = ~kinematic.cpu().numpy()
    obstacle_pts = truth_pos[0][~real]
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(10, 5))
    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, args.out, dpi=100):
        for t in range(args.steps):
            for ax, title, pts in ((ax_t, "truth", truth_pos[t + 1]), (ax_m, "model", model_pos[t])):
                ax.clear()
                ax.scatter(obstacle_pts[:, 0], obstacle_pts[:, 1], c="gray", s=8)
                ax.scatter(pts[real][:, 0], pts[real][:, 1], c="crimson", s=40)
                ax.set_title(title)
                ax.set_xlim(0, 40)
                ax.set_ylim(0, 40)
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    main()
