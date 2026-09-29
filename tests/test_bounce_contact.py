import numpy as np

import bounce


def test_ball_pair_forces_uses_per_ball_radius():
    xs = np.array([10.0, 11.0])
    ys = np.array([10.0, 10.0])
    radii = np.array([0.75, 0.5])  # sum 1.25 > distance 1.0 -> overlapping
    fx, fy = bounce.ball_pair_forces(xs, ys, radii, stiffness=400.0)
    assert fx[0] < 0.0 and fx[1] > 0.0  # pushed apart along x
    assert np.isclose(fx[0], -fx[1])
    assert np.all(fy == 0.0)


def test_ball_pair_forces_no_overlap_is_zero():
    xs = np.array([10.0, 20.0])
    ys = np.array([10.0, 10.0])
    radii = np.array([0.75, 0.75])
    fx, fy = bounce.ball_pair_forces(xs, ys, radii, stiffness=400.0)
    assert np.all(fx == 0.0) and np.all(fy == 0.0)


def test_wall_force_uses_per_ball_radius():
    xs = np.array([0.2, 0.2])
    ys = np.array([10.0, 10.0])
    radii = np.array([0.5, 1.0])  # ball 1 penetrates more (bigger radius)
    fx, fy = bounce.wall_force(xs, ys, n=20, radii=radii, stiffness=400.0)
    assert fx[0] > 0.0 and fx[1] > fx[0]  # both pushed right (+x), bigger ball harder


def test_wall_segment_force_pushes_off_middle_of_segment():
    # horizontal segment at y=10, x in [5, 15]; ball sitting just below it
    xs = np.array([10.0])
    ys = np.array([9.7])
    radii = np.array([0.5])
    fx, fy = bounce.wall_segment_force(xs, ys, "h", coord=10.0, lo=5.0, hi=15.0,
                                        radii=radii + 0.1, stiffness=400.0)
    assert fx[0] == 0.0 and fy[0] < 0.0  # pushed straight down, away from the segment


def test_wall_segment_force_near_endpoint_pushes_away_from_tip():
    # ball sitting just past the segment's right end (x=16), close to y=10
    xs = np.array([16.0])
    ys = np.array([10.0])
    radii = np.array([0.5])
    fx, fy = bounce.wall_segment_force(xs, ys, "h", coord=10.0, lo=5.0, hi=15.0,
                                        radii=radii + 1.2, stiffness=400.0)
    assert fx[0] > 0.0 and fy[0] == 0.0  # nearest point is the tip (15, 10); pushed +x, not off the line


def test_wall_segment_force_zero_far_away():
    xs = np.array([10.0])
    ys = np.array([2.0])
    radii = np.array([0.5])
    fx, fy = bounce.wall_segment_force(xs, ys, "h", coord=10.0, lo=5.0, hi=15.0,
                                        radii=radii, stiffness=400.0)
    assert fx[0] == 0.0 and fy[0] == 0.0


def test_integrate_skips_kinematic_body():
    balls = [{"x": 10.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "kinematic": True},
             {"x": 20.0, "y": 20.0, "vx": 0.0, "vy": 0.0}]
    forces = [(100.0, 100.0), (5.0, 0.0)]
    bounce.integrate(balls, forces, dt=0.1)
    assert balls[0]["x"] == 10.0 and balls[0]["y"] == 10.0 and balls[0]["vx"] == 0.0
    assert balls[1]["vx"] > 0.0 and balls[1]["x"] > 20.0


def test_integrate_divides_by_mass():
    light = [{"x": 10.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "mass": 0.5}]
    heavy = [{"x": 10.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "mass": 2.0}]
    bounce.integrate(light, [(1.0, 0.0)], dt=0.1)
    bounce.integrate(heavy, [(1.0, 0.0)], dt=0.1)
    assert light[0]["vx"] > heavy[0]["vx"] > 0.0


def test_compute_forces_backward_compatible_default():
    # no radius/mass/kinematic keys, no segments: same as before the change
    balls = bounce.init_balls(3, n=20, vy=2.0, rng=__import__("random").Random(4738))
    forces = bounce.compute_forces(balls, n=20, gravity=9.0, radius=0.75, stiffness=400.0)
    assert len(forces) == 3 and all(len(f) == 2 for f in forces)


def test_compute_forces_includes_obstacle_circle_as_kinematic_ball():
    real = {"x": 10.5, "y": 10.0, "vx": 0.0, "vy": 0.0, "radius": 0.75, "mass": 1.0}
    obstacle = {"x": 10.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "radius": 1.0, "kinematic": True}
    forces = bounce.compute_forces([real, obstacle], n=40, gravity=0.0, radius=0.75, stiffness=400.0)
    assert forces[0][0] > 0.0  # real ball pushed away (+x) from the obstacle at its left
    assert forces[1][0] == -forces[0][0]  # Newton's 3rd law: obstacle feels the exact opposite force (still won't move -- that's integrate's job, not compute_forces')


def test_compute_forces_with_wall_segment():
    balls = [{"x": 10.0, "y": 9.6, "vx": 0.0, "vy": 0.0, "radius": 0.5}]
    segments = [{"orientation": "h", "coord": 10.0, "lo": 5.0, "hi": 15.0, "radius": 0.1}]
    forces = bounce.compute_forces(balls, n=40, gravity=0.0, radius=0.75, stiffness=400.0, segments=segments)
    assert forces[0][1] < 0.0  # pushed down, away from the segment above it
