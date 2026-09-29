import random

import numpy as np
import torch

import bounce
from scripts import contact_truth as ct


def _cpu_rollout(balls, n, steps, dt, gravity, stiffness, substeps, segments):
    G = bounce.make_grid(n)
    balls = [dict(b) for b in balls]
    ps, vs = [np.array([[b["x"], b["y"]] for b in balls])], [np.array([[b["vx"], b["vy"]] for b in balls])]
    for _ in range(steps):
        bounce.step(G, n, balls, dt, gravity, bounce_default_radius(balls), stiffness, substeps, segments)
        ps.append(np.array([[b["x"], b["y"]] for b in balls]))
        vs.append(np.array([[b["vx"], b["vy"]] for b in balls]))
    return np.array(ps), np.array(vs)


def bounce_default_radius(balls):
    return 0.75  # every ball in these tests sets its own "radius"


def _make_balls(rng):
    return [
        {"x": 10.0, "y": 10.0, "vx": 1.0, "vy": 0.0, "radius": 0.75, "mass": 1.0},
        {"x": 12.0, "y": 10.3, "vx": -1.0, "vy": 0.2, "radius": 0.75, "mass": 2.0},
        {"x": 5.0, "y": 5.0, "vx": 0.0, "vy": 0.0, "radius": 1.0, "kinematic": True},
    ]


def test_gpu_truth_matches_cpu_reference_ball_ball_and_obstacle():
    rng = random.Random(4738)
    balls = _make_balls(rng)
    segments = [{"orientation": "h", "coord": 0.0, "lo": 0.0, "hi": 19.0, "radius": 0.0}]  # degenerate/no-op segment
    n, steps, dt, gravity, stiffness, substeps = 20, 30, 0.05, 9.0, 400.0, 8
    cpu_pos, cpu_vel = _cpu_rollout(balls, n, steps, dt, gravity, stiffness, substeps, segments)
    gpu_pos, gpu_vel = ct.rollout(balls, n, steps, dt, gravity, stiffness, substeps, segments, device="cpu")
    assert np.allclose(cpu_pos, gpu_pos, atol=1e-4)
    assert np.allclose(cpu_vel, gpu_vel, atol=1e-4)


def test_gpu_truth_matches_cpu_reference_with_real_wall_segment():
    balls = [{"x": 10.0, "y": 9.6, "vx": 0.0, "vy": -2.0, "radius": 0.5, "mass": 1.0}]
    segments = [{"orientation": "h", "coord": 10.0, "lo": 5.0, "hi": 15.0, "radius": 0.1}]
    n, steps, dt, gravity, stiffness, substeps = 40, 20, 0.05, 0.0, 400.0, 8
    cpu_pos, cpu_vel = _cpu_rollout(balls, n, steps, dt, gravity, stiffness, substeps, segments)
    gpu_pos, gpu_vel = ct.rollout(balls, n, steps, dt, gravity, stiffness, substeps, segments, device="cpu")
    assert np.allclose(cpu_pos, gpu_pos, atol=1e-4)
    assert np.allclose(cpu_vel, gpu_vel, atol=1e-4)
