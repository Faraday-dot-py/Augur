# CFD Contact/Collision Generalization (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `CentralForceDynamics` into a mass/radius/kinematic-aware `ContactForceDynamics` so one learned pairwise potential covers gravity, ball-ball contact, and ball-obstacle contact (arbitrary rigid shapes as boundary point clouds), Phase 1 scope only (no spin/friction).

**Architecture:** Per-body state grows to `(pos, vel, mass, radius, kinematic)`. A single learned potential `V(d, r_i+r_j, s_ij)` (s_ij a symmetric function of mass_i/mass_j) replaces `CentralForceDynamics`'s `f(d)`; force is `-dV/dd` along the pair line via autograd, integrated with per-mass acceleration and kinematic bodies skipping position/velocity updates. Obstacles (circles, axis-aligned finite wall segments) are injected as `kinematic=True` nodes into the same radius graph real bodies use — no new geometry subsystem. Truth stays exclusively `bounce.py`'s penalty-force physics, extended in place and ported to a GPU-vectorized torch version for data generation.

**Tech Stack:** PyTorch (model + GPU truth), NumPy (CPU reference truth in `bounce.py`), pytest, Polaris (SLURM) for training compute per [[infra_polaris_for_training]].

**Spec:** `docs/superpowers/specs/2026-09-28-cfd-contact-generalization-design.md`

## Global Constraints

- Truth physics is exclusively `bounce.py`'s existing penalty-force model (C1-smooth, symplectic-Euler), extended in place — no new/different truth engine (spec §6, §9).
- Obstacle primitives: circles and axis-aligned flat wall segments only, placed at arbitrary position; no arbitrary-angle walls, no polygons (spec §9).
- Force law is exactly `F_ij = -∇_d V(d, r_i+r_j, s_ij)`, central (along the pair line) — no free-direction force vector (spec §3, angular-momentum-conservation rationale).
- `radius=0`/no-obstacle/no-spin must reduce to today's `CentralForceDynamics` gravity behavior (spec §5) — this is a concrete parity requirement, not just a design intention.
- Kinematic bodies exert force normally but never receive a position/velocity update (spec §2).
- Spin and the tangential/friction force are Phase 2, deferred — do not implement in this plan (spec §0, §3).
- All compute (data generation, truth rollouts, training, eval) runs on Polaris GPU, never local CPU, per [[feedback_generate_data_on_gpu]]; only video rendering is local.
- Held-out generalization is evaluated on seeds 9000/12000, per [[feedback_select_on_heldout_seeds]].
- Standard training epoch/iteration budgets follow existing `train_gravity_dynamics.py`-style checkpointing (already saves each log interval, satisfying the periodic-checkpoint rule).

## Review Focus

- **Fast-moving body vs. sparse obstacle sampling**: a ball whose per-step displacement exceeds the gap between adjacent boundary points can tunnel through an obstacle in one step (spec §4 boundary-sampling question #1, same failure mode already on record for ball-ball contact). Test: a fast ball aimed at a coarsely-sampled obstacle does not pass through within one training-scale step.
- **Kinematic body receiving force**: a kinematic obstacle node must accumulate incoming force from real bodies (so it correctly pushes them back) but must never move itself, even under large forces. Test: obstacle position is bit-identical before/after a rollout under load.
- **Degenerate/zero-radius edge case**: `r_i + r_j == 0` (today's point-mass gravity case) must not divide-by-zero or NaN in the potential net's `r_sum` input or in `penalty_force`-style thresholds. Test: radius-0 rollout stays finite and matches `CentralForceDynamics` behavior at init.
- **Unequal mass at contact**: a heavy body and light body in contact must conserve total momentum (`sum m_i v_i`), not naive `sum v_i` — the existing conservation tests only cover unit mass. Test: mass-ratio pair-contact scene conserves `sum(m*v)` exactly, not `sum(v)`.
- **Wall-segment endpoints**: a ball approaching near a finite wall segment's end (not its middle) must see distance-to-nearest-point-on-segment (a 2D point-to-segment distance), not the infinite-line distance the box-boundary code uses today — otherwise a ball can clip through just past the segment's tip. Test: a ball positioned beyond a segment's `hi`/`lo` extent, offset perpendicular to it, gets a force directed away from the segment's endpoint, not straight off the line.

---

## File Structure

- **Modify `bounce.py`**: generalize `ball_pair_forces`/`wall_force` from scalar `radius` to per-ball `radii` arrays, add per-ball `mass`/`kinematic` handling in `integrate`, add `wall_segment_force` (finite axis-aligned segment, point-to-segment distance) and thread an optional `segments` list through `compute_forces`/`step`. All new parameters default to today's values, so every existing caller is unaffected.
- **Create `scripts/contact_truth.py`**: GPU-vectorized torch port of the same physics (mass/radius/kinematic/segments-aware), used for training-data generation at scale; a parity test checks it's bit-comparable (within float32 tolerance) to `bounce.py`'s CPU reference.
- **Create `model/obstacles.py`**: boundary-point sampling for circle and wall-segment obstacles, with a swept-density helper, used both by the training scene sampler and (later) by inference-time obstacle placement.
- **Create `model/contact_force.py`**: `ContactForceDynamics`, the mass/radius/kinematic-aware successor to `CentralForceDynamics`.
- **Create `tests/test_bounce_contact.py`**: CPU truth tests (per-ball radius/mass/kinematic, wall segments, backward compatibility).
- **Create `tests/test_contact_truth.py`**: GPU-vs-CPU truth parity test.
- **Create `tests/test_obstacles.py`**: boundary-point sampling tests.
- **Create `tests/test_contact_force.py`**: `ContactForceDynamics` conservation/kinematic/structural tests.
- **Create `scripts/train_contact_dynamics.py`**: scene sampler (ball-ball mass/radius sweep, circle-obstacle scenes, wall-segment scenes with swept boundary density) + training loop, following `scripts/train_gravity_dynamics.py`'s pattern.
- **Create `scripts/polaris_train_contact_v1.sh`**: Polaris SLURM job, smoke-scale first run.
- **Create `scripts/eval_contact_generalization.py`**: held-out obstacle shape/position/density and mass/radius-ratio generalization, plus an O(N)/tiling sanity check.

---

### Task 1: `bounce.py` — per-ball radius/mass/kinematic + wall-segment obstacles

**Files:**
- Modify: `bounce.py:94-176` (`wall_force`, `ball_pair_forces`, `compute_forces`, `integrate`, `step`)
- Test: `tests/test_bounce_contact.py`

**Interfaces:**
- Produces: `bounce.wall_force(xs, ys, n, radii, stiffness)` (was scalar `radius`), `bounce.ball_pair_forces(xs, ys, radii, stiffness)` (was scalar `radius`), `bounce.wall_segment_force(xs, ys, orientation, coord, lo, hi, radii, stiffness) -> (fx, fy)` (new), `bounce.compute_forces(balls, n, gravity, radius, stiffness, segments=None)`, `bounce.integrate(balls, forces, dt)`, `bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps, segments=None)`. A ball dict may now carry optional `"radius"` (float, overrides the `radius` arg for that ball), `"mass"` (float, default 1.0), `"kinematic"` (bool, default False). A segment is a dict `{"orientation": "h"|"v", "coord": float, "lo": float, "hi": float, "radius": float}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bounce_contact.py
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
                                        radii=radii + 0.2, stiffness=400.0)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_bounce_contact.py -v`
Expected: FAIL — `wall_segment_force` doesn't exist, `ball_pair_forces`/`wall_force` reject/mishandle array args as currently written, `compute_forces` has no `segments` param, `integrate` doesn't read `mass`/`kinematic`.

- [ ] **Step 3: Implement**

Replace `bounce.py:94-176` with:

```python
def wall_force(xs, ys, n, radii, stiffness):
    """Continuous force pushing each ball back once it overlaps a wall, in
    place of the old hard position-reflect. Walls are treated as immovable,
    so all of the force/energy goes into the ball. xs/ys/radii are numpy
    arrays, one entry per ball (radii lets obstacle/real balls differ)."""
    lo, hi = 0.0, n - 1.0
    left_x = (xs - lo) < radii
    right_x = ~left_x & ((hi - xs) < radii)
    fx = np.where(left_x, penalty_force(radii - (xs - lo), stiffness), 0.0)
    fx -= np.where(right_x, penalty_force(radii - (hi - xs), stiffness), 0.0)
    left_y = (ys - lo) < radii
    right_y = ~left_y & ((hi - ys) < radii)
    fy = np.where(left_y, penalty_force(radii - (ys - lo), stiffness), 0.0)
    fy -= np.where(right_y, penalty_force(radii - (hi - ys), stiffness), 0.0)
    return fx, fy


def wall_segment_force(xs, ys, orientation, coord, lo, hi, radii, stiffness):
    """Force from one finite, axis-aligned wall segment (e.g. an obstacle
    edge, not necessarily a box boundary) on every ball. The nearest point
    on the segment to a ball is the perpendicular foot when the ball's
    along-segment coordinate falls inside [lo, hi], and the nearer endpoint
    otherwise -- the same point-to-segment distance used for the finite
    extent, so a ball can't clip past a segment's tip (radii is the sum of
    each ball's own radius and the segment's radius, precomputed by the
    caller)."""
    if orientation == "h":
        along, perp = xs, ys
    else:
        along, perp = ys, xs
    nearest_along = np.clip(along, lo, hi)
    d_along = along - nearest_along
    d_perp = perp - coord
    dist = np.hypot(d_along, d_perp)
    close = dist < 1e-9
    safe_dist = np.where(close, 1.0, dist)
    n_along = np.where(close, 0.0, d_along / safe_dist)
    n_perp = np.where(close, 1.0, d_perp / safe_dist)
    f = penalty_force(radii - dist, stiffness)
    f_along, f_perp = f * n_along, f * n_perp
    if orientation == "h":
        return f_along, f_perp
    return f_perp, f_along


def ball_pair_forces(xs, ys, radii, stiffness):
    """Continuous repulsive force between every pair of overlapping balls,
    directed along their center line. `radii` is per-ball (obstacle circles
    and real balls can have different radii). Returns (fx, fy), each an
    (m, m) matrix where entry [i, j] is the force applied to ball j by ball
    i (antisymmetric: entry [j, i] == -entry [i, j], diagonal is zero)."""
    dx = xs.reshape(1, -1) - xs.reshape(-1, 1)
    dy = ys.reshape(1, -1) - ys.reshape(-1, 1)
    dist = np.hypot(dx, dy)
    close = dist < 1e-9
    safe_dist = np.where(close, 1.0, dist)
    nx = np.where(close, 1.0, dx / safe_dist)
    ny = np.where(close, 0.0, dy / safe_dist)
    dist = np.where(close, 0.0, dist)
    pair_radii = radii.reshape(1, -1) + radii.reshape(-1, 1)
    f = penalty_force(pair_radii - dist, stiffness)
    np.fill_diagonal(f, 0.0)
    return f * nx, f * ny


def compute_forces(balls, n, gravity, radius, stiffness, segments=None):
    """Net force on each ball this step: gravity + wall contact + pairwise
    ball contact + any wall-segment obstacles. `radius`/`stiffness` are the
    defaults for balls without their own "radius" key (obstacle balls
    normally set their own). Force is no longer == acceleration once a ball
    has mass != 1 -- `integrate` divides by mass."""
    m = len(balls)
    if m == 0:
        return []
    xs = np.array([b["x"] for b in balls])
    ys = np.array([b["y"] for b in balls])
    radii = np.array([b.get("radius", radius) for b in balls])
    wfx, wfy = wall_force(xs, ys, n, radii, stiffness)
    base_x = gravity + wfx
    base_y = wfy
    if segments:
        for seg in segments:
            seg_radii = radii + seg["radius"]
            sfx, sfy = wall_segment_force(xs, ys, seg["orientation"], seg["coord"], seg["lo"], seg["hi"],
                                           seg_radii, stiffness)
            base_x = base_x + sfx
            base_y = base_y + sfy
    if m > 1:
        pfx, pfy = ball_pair_forces(xs, ys, radii, stiffness)
        # this system is numerically chaotic (stiff contacts), so matching
        # the original loop's exact left-to-right float accumulation order
        # (base term first, then each pairwise contribution in ascending
        # ball-index order) matters, not just the summation's math value --
        # cumsum (unlike .sum, which uses pairwise summation for long rows)
        # preserves that order
        chain_x = np.concatenate([base_x.reshape(-1, 1), -pfx], axis=1)
        chain_y = np.concatenate([base_y.reshape(-1, 1), -pfy], axis=1)
        fx = np.cumsum(chain_x, axis=1)[:, -1]
        fy = np.cumsum(chain_y, axis=1)[:, -1]
    else:
        fx, fy = base_x, base_y
    return np.stack([fx, fy], axis=1).tolist()


def integrate(balls, forces, dt):
    """Semi-implicit (symplectic) Euler: velocity updated from force/mass
    first, then position updated from the new velocity. A kinematic body
    (obstacle) still receives a force above but skips this update entirely
    -- it exerts force without ever being moved by it."""
    for state, (fx, fy) in zip(balls, forces):
        if state.get("kinematic", False):
            continue
        m = state.get("mass", 1.0)
        state["vx"] += fx / m * dt
        state["vy"] += fy / m * dt
        state["x"] += state["vx"] * dt
        state["y"] += state["vy"] * dt


def step(G, n, balls, dt, gravity, radius, stiffness, substeps, segments=None):
    """Advance by dt total, but in `substeps` smaller physics steps: the
    penalty force is stiff enough that symplectic Euler needs a finer
    resolution than one step per render frame to stay stable."""
    sub_dt = dt / substeps
    for _ in range(substeps):
        forces = compute_forces(balls, n, gravity, radius, stiffness, segments)
        integrate(balls, forces, sub_dt)
    splat_all(G, n, balls, radius)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_bounce_contact.py -v`
Expected: PASS (all 11 tests)

- [ ] **Step 5: Run the full existing suite to check for regressions**

Run: `PYTHONPATH=. pytest tests/ -k "not contact_force and not contact_truth and not obstacles" -v`
Expected: PASS — `main()`'s existing calls to `wall_force`/`ball_pair_forces`/`compute_forces`/`step` pass a scalar `radius`, but since every ball lacks a `"radius"` key, `radii = np.array([b.get("radius", radius) for b in balls])` reduces to the old scalar broadcast, and `integrate` with no `"mass"`/`"kinematic"` keys behaves exactly as before.

- [ ] **Step 6: Commit**

```bash
git add bounce.py tests/test_bounce_contact.py
git commit -m "bounce.py: per-ball radius/mass/kinematic + wall-segment obstacles (CFD contact-gen Phase 1, task 1)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

### Task 2: GPU-vectorized truth port (`scripts/contact_truth.py`)

**Files:**
- Create: `scripts/contact_truth.py`
- Test: `tests/test_contact_truth.py`

**Interfaces:**
- Consumes: `bounce.wall_force`, `bounce.wall_segment_force`, `bounce.ball_pair_forces`, `bounce.compute_forces`, `bounce.integrate`, `bounce.step` (Task 1, CPU reference for parity).
- Produces: `contact_truth.forces(p, radii, n, gravity, stiffness, segments=None) -> torch.Tensor (m, 2)` (raw force, not acceleration or mass/kinematic-adjusted — `rollout` applies mass division and the kinematic freeze), `contact_truth.rollout(balls, n, steps, dt, gravity, stiffness, substeps, segments, device) -> (pos (steps+1, m, 2) ndarray, vel (steps+1, m, 2) ndarray)`. `balls` is a list of dicts identical in shape to `bounce.py`'s (may include `"radius"`, `"mass"`, `"kinematic"`). `segments` is the same list-of-dicts shape as `bounce.compute_forces`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contact_truth.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_contact_truth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.contact_truth'`

- [ ] **Step 3: Implement**

```python
# scripts/contact_truth.py
"""GPU-vectorized torch port of bounce.py's penalty-force physics (per-ball
radius/mass/kinematic, plus wall-segment obstacles), for generating Phase 1
contact-generalization training data at scale. Must stay bit-comparable
(within float precision) with bounce.py's CPU reference -- see
tests/test_contact_truth.py -- since bounce.py is the one source of truth
(spec docs/superpowers/specs/2026-09-28-cfd-contact-generalization-design.md §6/§9).

Usage: PYTHONPATH=. python3 -c "from scripts import contact_truth" (library module, no CLI)
"""
import torch


def _penalty_force(penetration, stiffness):
    return torch.where(penetration > 0.0, stiffness * penetration * penetration, torch.zeros_like(penetration))


def _wall_force(xs, ys, n, radii, stiffness):
    lo, hi = 0.0, n - 1.0
    left_x = (xs - lo) < radii
    right_x = (~left_x) & ((hi - xs) < radii)
    fx = torch.where(left_x, _penalty_force(radii - (xs - lo), stiffness), torch.zeros_like(xs))
    fx = fx - torch.where(right_x, _penalty_force(radii - (hi - xs), stiffness), torch.zeros_like(xs))
    left_y = (ys - lo) < radii
    right_y = (~left_y) & ((hi - ys) < radii)
    fy = torch.where(left_y, _penalty_force(radii - (ys - lo), stiffness), torch.zeros_like(ys))
    fy = fy - torch.where(right_y, _penalty_force(radii - (hi - ys), stiffness), torch.zeros_like(ys))
    return fx, fy


def _wall_segment_force(xs, ys, orientation, coord, lo, hi, radii, stiffness):
    if orientation == "h":
        along, perp = xs, ys
    else:
        along, perp = ys, xs
    nearest_along = along.clamp(min=lo, max=hi)
    d_along = along - nearest_along
    d_perp = perp - coord
    dist = torch.hypot(d_along, d_perp)
    close = dist < 1e-9
    safe_dist = torch.where(close, torch.ones_like(dist), dist)
    n_along = torch.where(close, torch.zeros_like(dist), d_along / safe_dist)
    n_perp = torch.where(close, torch.ones_like(dist), d_perp / safe_dist)
    f = _penalty_force(radii - dist, stiffness)
    f_along, f_perp = f * n_along, f * n_perp
    if orientation == "h":
        return f_along, f_perp
    return f_perp, f_along


def _ball_pair_forces(xs, ys, radii, stiffness):
    dx = xs.unsqueeze(0) - xs.unsqueeze(1)
    dy = ys.unsqueeze(0) - ys.unsqueeze(1)
    dist = torch.hypot(dx, dy)
    close = dist < 1e-9
    safe_dist = torch.where(close, torch.ones_like(dist), dist)
    nx = torch.where(close, torch.ones_like(dist), dx / safe_dist)
    ny = torch.where(close, torch.zeros_like(dist), dy / safe_dist)
    dist = torch.where(close, torch.zeros_like(dist), dist)
    pair_radii = radii.unsqueeze(0) + radii.unsqueeze(1)
    f = _penalty_force(pair_radii - dist, stiffness)
    f.fill_diagonal_(0.0)
    return f * nx, f * ny


def forces(p, radii, n, gravity, stiffness, segments=None):
    """Net force (not yet divided by mass) on each body: gravity + walls +
    obstacle segments + pairwise ball contact. `p` is (m, 2), `radii` (m,)."""
    xs, ys = p[:, 0], p[:, 1]
    wfx, wfy = _wall_force(xs, ys, n, radii, stiffness)
    base_x = gravity + wfx
    base_y = wfy
    for seg in (segments or []):
        seg_radii = radii + seg["radius"]
        sfx, sfy = _wall_segment_force(xs, ys, seg["orientation"], seg["coord"], seg["lo"], seg["hi"],
                                        seg_radii, stiffness)
        base_x = base_x + sfx
        base_y = base_y + sfy
    if p.shape[0] > 1:
        pfx, pfy = _ball_pair_forces(xs, ys, radii, stiffness)
        pfx.fill_diagonal_(0.0)
        fx = base_x - pfx.sum(dim=1)
        fy = base_y - pfy.sum(dim=1)
    else:
        fx, fy = base_x, base_y
    return torch.stack([fx, fy], dim=1)


def rollout(balls, n, steps, dt, gravity, stiffness, substeps, segments, device):
    """Symplectic-Euler rollout matching bounce.py's step()/integrate(),
    with per-ball mass division and kinematic bodies frozen. Returns
    (pos, vel) as (steps+1, m, 2) numpy arrays, dtype float64 to match the
    CPU reference's accumulation precision closely enough for parity
    (see tests/test_contact_truth.py)."""
    m = len(balls)
    p = torch.tensor([[b["x"], b["y"]] for b in balls], dtype=torch.float64, device=device)
    v = torch.tensor([[b["vx"], b["vy"]] for b in balls], dtype=torch.float64, device=device)
    radii = torch.tensor([b.get("radius", 0.75) for b in balls], dtype=torch.float64, device=device)
    masses = torch.tensor([b.get("mass", 1.0) for b in balls], dtype=torch.float64, device=device)
    kinematic = torch.tensor([b.get("kinematic", False) for b in balls], dtype=torch.bool, device=device)
    sub_dt = dt / substeps
    ps, vs = [p.cpu().numpy().copy()], [v.cpu().numpy().copy()]
    for _ in range(steps):
        for _ in range(substeps):
            f = forces(p, radii, n, gravity, stiffness, segments)
            new_v = v + (f / masses.unsqueeze(-1)) * sub_dt
            v = torch.where(kinematic.unsqueeze(-1), v, new_v)
            new_p = p + v * sub_dt
            p = torch.where(kinematic.unsqueeze(-1), p, new_p)
        ps.append(p.cpu().numpy().copy())
        vs.append(v.cpu().numpy().copy())
    return __import__("numpy").array(ps), __import__("numpy").array(vs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_contact_truth.py -v`
Expected: PASS. If it fails on floating-point drift over 20-30 steps, tighten the CPU reference helper (`_cpu_rollout`) to call `bounce.step` with `substeps` exactly matching, and loosen `atol` only as a last resort (chaotic stiff-contact systems can diverge in exact float ordering — same caveat `bounce.py:142-147`'s comment already documents — so prefer fewer steps/no simultaneous multi-contact in the test scenes over a looser tolerance).

- [ ] **Step 5: Commit**

```bash
git add scripts/contact_truth.py tests/test_contact_truth.py
git commit -m "GPU-vectorized contact truth port (bit-parity with bounce.py), CFD contact-gen Phase 1 task 2

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

### Task 3: Obstacle boundary-point sampling (`model/obstacles.py`)

**Files:**
- Create: `model/obstacles.py`
- Test: `tests/test_obstacles.py`

**Interfaces:**
- Produces: `obstacles.sample_circle_boundary(cx, cy, radius, spacing, point_radius) -> (points (k,2) ndarray, radii (k,) ndarray)`, `obstacles.sample_wall_segment_boundary(orientation, coord, lo, hi, spacing, point_radius) -> (points (k,2) ndarray, radii (k,) ndarray)`, `obstacles.default_point_radius(spacing, overlap=1.25) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obstacles.py
import math

import numpy as np

from model import obstacles as ob


def test_default_point_radius_scales_with_spacing():
    assert ob.default_point_radius(1.0) == 0.625
    assert ob.default_point_radius(2.0) == 1.25


def test_sample_circle_boundary_points_lie_on_circle():
    pts, radii = ob.sample_circle_boundary(cx=5.0, cy=-3.0, radius=4.0, spacing=1.0, point_radius=0.7)
    d = np.hypot(pts[:, 0] - 5.0, pts[:, 1] - (-3.0))
    assert np.allclose(d, 4.0, atol=1e-9)
    assert np.all(radii == 0.7)
    assert len(pts) >= 3


def test_sample_circle_boundary_denser_spacing_gives_more_points():
    coarse, _ = ob.sample_circle_boundary(0.0, 0.0, 10.0, spacing=5.0, point_radius=1.0)
    fine, _ = ob.sample_circle_boundary(0.0, 0.0, 10.0, spacing=1.0, point_radius=1.0)
    assert len(fine) > len(coarse)


def test_sample_circle_boundary_adjacent_points_overlap():
    pts, radii = ob.sample_circle_boundary(0.0, 0.0, 10.0, spacing=2.0, point_radius=ob.default_point_radius(2.0))
    gap = np.hypot(*(pts[0] - pts[1]))
    assert gap < radii[0] + radii[1]  # adjacent disks overlap: no ball-sized hole in the surface


def test_sample_wall_segment_boundary_endpoints_included():
    pts, radii = ob.sample_wall_segment_boundary("h", coord=10.0, lo=5.0, hi=15.0, spacing=2.0, point_radius=0.5)
    assert np.isclose(pts[0, 0], 5.0) and np.isclose(pts[-1, 0], 15.0)
    assert np.all(pts[:, 1] == 10.0)
    assert np.all(radii == 0.5)


def test_sample_wall_segment_boundary_vertical():
    pts, _ = ob.sample_wall_segment_boundary("v", coord=3.0, lo=0.0, hi=10.0, spacing=5.0, point_radius=0.5)
    assert np.all(pts[:, 0] == 3.0)
    assert np.isclose(pts[0, 1], 0.0) and np.isclose(pts[-1, 1], 10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_obstacles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.obstacles'`

- [ ] **Step 3: Implement**

```python
# model/obstacles.py
"""Boundary-point sampling for Phase 1 obstacle primitives (circles, finite
axis-aligned wall segments) -- turning a geometric obstacle into the finite
set of kinematic point samples the model actually sees, per spec §4's
"boundary-sampling question" (density / per-point radius / swept density
are one joint tuning setting, not independent choices)."""
import math

import numpy as np


def default_point_radius(spacing, overlap=1.25):
    """Point radius that keeps adjacent boundary samples' contact disks
    overlapping by `overlap`x half-spacing, so the sampled surface has no
    ball-sized gaps (spec §4 continuity requirement)."""
    return 0.5 * spacing * overlap


def sample_circle_boundary(cx, cy, radius, spacing, point_radius):
    """Evenly spaced points along a circle's boundary at ~`spacing`
    arc-length, each carrying `point_radius` as its own contact radius."""
    circumference = 2 * math.pi * radius
    k = max(3, round(circumference / spacing))
    theta = np.linspace(0.0, 2 * math.pi, k, endpoint=False)
    pts = np.stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)], axis=1)
    return pts, np.full(k, point_radius)


def sample_wall_segment_boundary(orientation, coord, lo, hi, spacing, point_radius):
    """Points along a finite axis-aligned wall segment at ~`spacing`,
    always including both endpoints (so the sampled surface doesn't fall
    short of the segment's actual extent)."""
    k = max(2, round((hi - lo) / spacing) + 1)
    t = np.linspace(lo, hi, k)
    if orientation == "h":
        pts = np.stack([t, np.full(k, coord)], axis=1)
    else:
        pts = np.stack([np.full(k, coord), t], axis=1)
    return pts, np.full(k, point_radius)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_obstacles.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/obstacles.py tests/test_obstacles.py
git commit -m "obstacle boundary-point sampling (circles + wall segments), CFD contact-gen Phase 1 task 3

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

### Task 4: `ContactForceDynamics` model (`model/contact_force.py`)

**Files:**
- Create: `model/contact_force.py`
- Test: `tests/test_contact_force.py`

**Interfaces:**
- Consumes: `model.token_graph.build_radius_graph_cells(positions, neighbor_radius) -> edges (2, E) long tensor` (unchanged, Task 1's spec §2 requirement to reuse it as-is).
- Produces: `ContactForceDynamics(dt=0.1, neighbor_radius=100.0, width=64)`, method `pair_features(d, r_sum, mass_src, mass_dst) -> (E, 5) tensor` (`[log(d), r_sum, m_sum, m_prod, m_diff]`), method `accel(positions, radius, mass) -> (N, 2) tensor`, `forward(positions, velocities, hidden, radius, mass, kinematic) -> (dp, dv, hidden)` where `radius`/`mass` are `(N,)` float tensors and `kinematic` is an `(N,)` bool tensor.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_contact_force.py
import torch

from model.contact_force import ContactForceDynamics


def _dyn():
    # random-weight fixture for the momentum/kinematic tests below, matching
    # tests/test_central_force.py::test_momentum_conserved_by_construction's
    # pattern -- do NOT re-zero the final layer here (that's only for the
    # separate "zero-init is free flight" test): re-zeroing it after
    # randomizing everything else makes V identically 0 (zero weight matrix
    # kills the last layer's output regardless of what the Tanh layers
    # upstream computed), producing a dead network where dv is always 0 --
    # a bug caught during Task 4's implementation.
    torch.manual_seed(4738)
    dyn = ContactForceDynamics()
    for p in dyn.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return dyn


def test_pair_features_is_swap_symmetric_in_mass():
    dyn = ContactForceDynamics()
    d = torch.tensor([[2.0], [2.0]])
    r_sum = torch.tensor([[0.5], [0.5]])
    ab = dyn.pair_features(d, r_sum, torch.tensor([[1.0], [3.0]]), torch.tensor([[3.0], [1.0]]))
    assert torch.allclose(ab[0], ab[1])  # m_i, m_j swapped gives the same feature vector


def test_zero_init_is_free_flight():
    dyn = ContactForceDynamics()
    pos = torch.rand(5, 2) * 10
    vel = torch.randn(5, 2)
    radius = torch.full((5,), 0.75)
    mass = torch.ones(5)
    kinematic = torch.zeros(5, dtype=torch.bool)
    dp, dv, _ = dyn(pos, vel, torch.zeros(5, 1), radius, mass, kinematic)
    assert torch.all(dp == 0) and torch.all(dv == 0)


def test_momentum_conserved_with_unequal_mass():
    dyn = _dyn()
    # third body must be outside ContactForceDynamics' default
    # neighbor_radius=100.0 to be graph-disconnected and genuinely
    # unaffected -- (30, 30) is only ~28 units from the pair, well inside
    # that cutoff, and was a second bug caught during Task 4's
    # implementation (dv[2] was not actually zero: the body was connected
    # and felt a small random-potential force).
    pos = torch.tensor([[10.0, 10.0], [10.5, 10.0], [1000.0, 1000.0]])
    vel = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    radius = torch.tensor([0.75, 0.75, 0.75])
    mass = torch.tensor([1.0, 3.0, 1.0])
    kinematic = torch.zeros(3, dtype=torch.bool)
    dp, dv, _ = dyn(pos, vel, torch.zeros(3, 1), radius, mass, kinematic)
    p_before = (mass.unsqueeze(-1) * vel).sum(0)
    p_after = (mass.unsqueeze(-1) * (vel + dv)).sum(0)
    assert torch.allclose(p_before, p_after, atol=1e-4)
    assert torch.allclose(dv[2], torch.zeros(2), atol=1e-6)  # far body unaffected


def test_kinematic_body_never_moves():
    dyn = _dyn()
    pos = torch.tensor([[10.0, 10.0], [10.5, 10.0]])
    vel = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
    radius = torch.tensor([0.75, 1.0])
    mass = torch.tensor([1.0, 1.0])
    kinematic = torch.tensor([False, True])
    dp, dv, _ = dyn(pos, vel, torch.zeros(2, 1), radius, mass, kinematic)
    assert dp[1, 0] == 0.0 and dp[1, 1] == 0.0
    assert dv[1, 0] == 0.0 and dv[1, 1] == 0.0
    assert dv[0].abs().sum() > 0  # the free body does feel the obstacle


def test_radius_zero_is_finite():
    dyn = _dyn()
    pos = torch.tensor([[10.0, 10.0], [10.0001, 10.0]])
    vel = torch.zeros(2, 2)
    radius = torch.zeros(2)
    mass = torch.ones(2)
    kinematic = torch.zeros(2, dtype=torch.bool)
    dp, dv, _ = dyn(pos, vel, torch.zeros(2, 1), radius, mass, kinematic)
    assert torch.isfinite(dp).all() and torch.isfinite(dv).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_contact_force.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.contact_force'`

- [ ] **Step 3: Implement**

```python
# model/contact_force.py
import torch
import torch.nn as nn

from model.token_graph import build_radius_graph_cells


class ContactForceDynamics(nn.Module):
    """Learned pairwise potential V(d, r_i+r_j, s_ij), s_ij a swap-symmetric
    function of (mass_i, mass_j). Force is -dV/dd along the pair line (via
    autograd), so momentum is conserved exactly and a step is symplectic
    (same structure as CentralForceDynamics, generalized per
    docs/superpowers/specs/2026-09-28-cfd-contact-generalization-design.md
    §3). radius=0 for every body reduces this to CentralForceDynamics'
    f(d) exactly (spec §5) since r_sum/m_sum/m_prod/m_diff are then extra,
    learnable-away input channels rather than a structural change.
    kinematic bodies (obstacles) still exert force but never receive a
    position/velocity update (spec §2). Same call convention as
    CentralForceDynamics/TokenFreeDynamics otherwise: returns (dp, dv,
    hidden) for `pos + vel * dt + dp`, `vel + dv`."""

    def __init__(self, dt=0.1, neighbor_radius=100.0, width=64):
        super().__init__()
        self.dt = dt
        self.neighbor_radius = neighbor_radius
        self.hidden_dim = 1
        self.potential = nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                        nn.Linear(width, 1))
        nn.init.zeros_(self.potential[-1].weight)
        nn.init.zeros_(self.potential[-1].bias)

    def pair_features(self, d, r_sum, mass_src, mass_dst):
        m_sum = mass_src + mass_dst
        m_prod = mass_src * mass_dst
        m_diff = (mass_src - mass_dst).abs()
        return torch.cat([torch.log(d), r_sum, m_sum, m_prod, m_diff], dim=-1)

    def accel(self, positions, radius, mass):
        edges = build_radius_graph_cells(positions, self.neighbor_radius)
        acc = torch.zeros_like(positions)
        if edges.shape[1] == 0:
            return acc
        src, dst = edges[0], edges[1]
        rel = positions[src] - positions[dst]
        with torch.enable_grad():
            d = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
            d.requires_grad_(True)
            r_sum = (radius[src] + radius[dst]).unsqueeze(-1)
            feat = self.pair_features(d, r_sum, mass[src].unsqueeze(-1), mass[dst].unsqueeze(-1))
            v = self.potential(feat)
            (grad_d,) = torch.autograd.grad(v.sum(), d, create_graph=True)
        force = grad_d * rel / d
        net_force = acc.index_add(0, dst, force)
        return net_force / mass.unsqueeze(-1)

    def forward(self, positions, velocities, hidden, radius, mass, kinematic):
        dt = self.dt
        kin = kinematic.unsqueeze(-1)
        a0 = self.accel(positions, radius, mass)
        dp = torch.where(kin, torch.zeros_like(a0), 0.5 * dt * dt * a0)
        a1 = self.accel(positions + velocities * dt + dp, radius, mass)
        dv = 0.5 * dt * (a0 + a1)
        dv = torch.where(kin, torch.zeros_like(dv), dv)
        return dp, dv, hidden
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_contact_force.py -v`
Expected: PASS. `torch.enable_grad()` inside `accel` makes the `torch.autograd.grad(v.sum(), d, create_graph=True)` call work even when a caller (a `with torch.no_grad():` eval loop) has disabled grad globally; the returned `force` stays part of a live graph (harmless when a caller never calls `.backward()`, needed when the training loop does).

- [ ] **Step 5: Commit**

```bash
git add model/contact_force.py tests/test_contact_force.py
git commit -m "ContactForceDynamics: mass/radius/kinematic-aware learned pair potential, CFD contact-gen Phase 1 task 4

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

### Task 5: Scene sampler + training script (`scripts/train_contact_dynamics.py`)

**Files:**
- Create: `scripts/train_contact_dynamics.py`

**Interfaces:**
- Consumes: `scripts.contact_truth.rollout` (Task 2), `model.obstacles.sample_circle_boundary`/`sample_wall_segment_boundary`/`default_point_radius` (Task 3), `model.contact_force.ContactForceDynamics` (Task 4).
- Produces: `train_contact_dynamics.make_scene(rng, min_bodies, max_bodies) -> dict` (a scene spec: ball count, per-ball mass/radius draw, optional obstacle), `train_contact_dynamics.build_truth(scene, steps, dt, gravity, stiffness, substeps) -> (balls, pos, vel, radius, mass, kinematic)` tensors ready for training, `main()` CLI following `scripts/train_gravity_dynamics.py`'s `--iters`/`--batch`/`--k-start`/`--k-end`/`--checkpoint`/`--out` convention.

- [ ] **Step 1: Write the script** (no unit test for the training loop itself — parity/conservation are already covered by Tasks 1-4; this task's own correctness gate is a short smoke run in Step 2)

```python
# scripts/train_contact_dynamics.py
"""Phase 1 CFD contact-generalization training: ball-ball contact (varying
mass/radius), ball-vs-circle-obstacle, and ball-vs-wall-segment-obstacle
scenes, truth from scripts/contact_truth.py (bounce.py's own physics, GPU
vectorized), model from model/contact_force.py. Boundary-point density is
swept per scene (spec §4/§6/§9 resolved decision), not fixed.

Usage: PYTHONPATH=. python3 -m scripts.train_contact_dynamics --device cuda
"""
import argparse
import json
import random

import numpy as np
import torch

from model import obstacles as ob
from model.contact_force import ContactForceDynamics
from scripts import contact_truth as ct

MASS_RANGE = (0.25, 4.0)  # train range per docs/debugging/z-and-mass-channels-feasibility.md recommendation
RADIUS_RANGE = (0.4, 1.2)
SPACING_RANGE = (0.6, 2.0)  # swept boundary-point spacing (spec §4 boundary-sampling question)


def make_scene(rng, min_bodies, max_bodies, n):
    """One training scene: a handful of real balls, plus (with some
    probability) one obstacle -- a circle or a finite wall segment,
    injected as extra kinematic point-cloud entries in the same ball list
    scripts/contact_truth.py already understands."""
    num_real = rng.randint(min_bodies, max_bodies)
    balls = []
    for _ in range(num_real):
        r = rng.uniform(*RADIUS_RANGE)
        m = rng.uniform(*MASS_RANGE)
        balls.append({
            "x": rng.uniform(2 * r, n - 1 - 2 * r), "y": rng.uniform(2 * r, n - 1 - 2 * r),
            "vx": rng.uniform(-3.0, 3.0), "vy": rng.uniform(-3.0, 3.0),
            "radius": r, "mass": m,
        })
    kind = rng.choice(["none", "circle", "wall"])
    spacing = rng.uniform(*SPACING_RANGE)
    point_radius = ob.default_point_radius(spacing)
    if kind == "circle":
        cx, cy, cr = rng.uniform(0.2 * n, 0.8 * n), rng.uniform(0.2 * n, 0.8 * n), rng.uniform(2.0, 6.0)
        pts, radii = ob.sample_circle_boundary(cx, cy, cr, spacing, point_radius)
        for (x, y), r in zip(pts, radii):
            balls.append({"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "radius": float(r), "kinematic": True})
    elif kind == "wall":
        orientation = rng.choice(["h", "v"])
        coord = rng.uniform(0.3 * n, 0.7 * n)
        lo = rng.uniform(0.1 * n, 0.4 * n)
        hi = lo + rng.uniform(0.2 * n, 0.4 * n)
        pts, radii = ob.sample_wall_segment_boundary(orientation, coord, lo, hi, spacing, point_radius)
        for (x, y), r in zip(pts, radii):
            balls.append({"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "radius": float(r), "kinematic": True})
    return balls


def build_truth(balls, steps, dt, gravity, stiffness, substeps, device):
    pos, vel = ct.rollout(balls, n=1000, steps=steps, dt=dt, gravity=gravity, stiffness=stiffness,
                           substeps=substeps, segments=None, device=device)
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
    ap.add_argument("--gravity", type=float, default=9.0)
    ap.add_argument("--stiffness", type=float, default=400.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--neighbor-radius", type=float, default=6.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--checkpoint", default="checkpoints/contact_dynamics_v1.pt")
    ap.add_argument("--out", default="results/contact_dynamics_v1.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dyn = ContactForceDynamics(dt=args.dt, neighbor_radius=args.neighbor_radius).to(args.device)
    opt = torch.optim.Adam(dyn.parameters(), lr=args.lr)
    for it in range(args.iters):
        k = int(round(args.k_start + (args.k_end - args.k_start) * it / max(1, args.iters - 1)))
        opt.zero_grad()
        total = 0.0
        for _ in range(args.batch):
            balls = make_scene(rng, args.min_bodies, args.max_bodies, n=40)
            pos, vel, radius, mass, kinematic = build_truth(balls, args.steps, args.dt, args.gravity,
                                                              args.stiffness, args.substeps, args.device)
            t0 = rng.randint(0, args.steps - k)
            ps, vs = unroll(dyn, pos[t0].clone(), vel[t0].clone(), radius, mass, kinematic, k, args.dt)
            tp = torch.tensor(pos[t0 + 1:t0 + k + 1], dtype=torch.float32, device=args.device)
            tv = torch.tensor(vel[t0 + 1:t0 + k + 1], dtype=torch.float32, device=args.device)
            loss = (((ps - tp) ** 2).mean() + 0.1 * ((vs - tv) ** 2).mean()) / args.batch
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        if it % args.log_every == 0:
            print(f"it {it} k {k} loss {total:.6f}", flush=True)
            torch.save(dyn.state_dict(), args.checkpoint)
    torch.save(dyn.state_dict(), args.checkpoint)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args)}, f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script on CPU with a tiny budget (correctness gate for this task)**

Run: `PYTHONPATH=. python3 -m scripts.train_contact_dynamics --device cpu --iters 5 --batch 2 --steps 8 --k-start 2 --k-end 4 --checkpoint /tmp/contact_smoke.pt --out /tmp/contact_smoke.json`
Expected: exits 0, prints 1 loss line, `loss` is finite (not NaN/inf), `/tmp/contact_smoke.pt` exists.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_contact_dynamics.py
git commit -m "contact-dynamics scene sampler + training loop (ball-ball, circle, wall-segment obstacles), CFD contact-gen Phase 1 task 5

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

### Task 6: Polaris training job (`scripts/polaris_train_contact_v1.sh`)

**Files:**
- Create: `scripts/polaris_train_contact_v1.sh`

**Interfaces:**
- Consumes: `scripts/train_contact_dynamics.py` (Task 5).
- Produces: `checkpoints/contact_dynamics_v1.pt`, `results/contact_dynamics_v1.json` on Polaris, downloaded locally afterward.

- [ ] **Step 1: Write the job script**

```bash
#!/bin/bash
#SBATCH --job-name=bounce-contact-gen-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=bounce-contact-gen-v1-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] contact-gen Phase 1 training"
python -m scripts.train_contact_dynamics --device cuda --seed 4738 --min-bodies 2 --max-bodies 6 \
  --iters 2000 --batch 8 --steps 30 --k-start 4 --k-end 16 --log-every 50 \
  --checkpoint checkpoints/contact_dynamics_v1.pt --out results/contact_dynamics_v1.json
echo "[$(date -Iseconds)] done"
```

Per [[reference_polaris_job_logistics]]: one GPU job at a time, submit via the Polaris MCP tools, real log path is `~/<name>-<id>.log` (not the `-%j.log` SBATCH placeholder path), model flags in this script must match whatever `scripts/eval_contact_generalization.py` (Task 7) loads with.

- [ ] **Step 2: Submit and confirm completion**

Run (via `mcp__polaris__polaris_submit_job` with this script): monitor via `mcp__polaris__polaris_job_status`/`polaris_job_logs` until it exits 0; then `mcp__polaris__polaris_download` `checkpoints/contact_dynamics_v1.pt` and `results/contact_dynamics_v1.json` to the local `checkpoints/`/`results/` directories.
Expected: log shows periodic `it <n> k <k> loss <finite>` lines with loss trending down; job exits 0.

- [ ] **Step 3: Commit the job script (not the downloaded checkpoint/results, which are gitignored artifacts per existing convention — check `.gitignore` for `checkpoints/`/`results/` before adding)**

```bash
git add scripts/polaris_train_contact_v1.sh
git commit -m "Polaris job: Phase 1 contact-gen training run v1

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

### Task 7: Generalization eval (`scripts/eval_contact_generalization.py`)

**Files:**
- Create: `scripts/eval_contact_generalization.py`

**Interfaces:**
- Consumes: `checkpoints/contact_dynamics_v1.pt` (Task 6), `model.contact_force.ContactForceDynamics`, `model.obstacles`, `scripts.contact_truth.rollout`, `model.token_graph.build_radius_graph_cells`.
- Produces: `results/contact_generalization_v1.json` (per-scenario err@5/10/20 + momentum/energy diagnostics), printed summary table.

- [ ] **Step 1: Write the eval script**

```python
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
from model.contact_force import ContactForceDynamics
from model.token_graph import build_radius_graph_cells
from scripts import contact_truth as ct
from scripts.train_contact_dynamics import unroll


def scenario_unseen_obstacle_shape(rng, n=40):
    """Held-out: a much bigger circle at a swept density outside training's
    SPACING_RANGE (0.6-2.0), and a ball aimed straight at it -- the actual
    target capability (spec §0/§7): drop a shape in after training."""
    spacing, point_radius = 3.0, ob.default_point_radius(3.0)
    pts, radii = ob.sample_circle_boundary(20.0, 20.0, 10.0, spacing, point_radius)
    balls = [{"x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "radius": float(r), "kinematic": True}
             for (x, y), r in zip(pts, radii)]
    balls.append({"x": 5.0, "y": 20.0, "vx": 4.0, "vy": 0.0, "radius": 0.75, "mass": 1.0})
    return balls


def scenario_unseen_mass_ratio(rng, n=40):
    """Held-out: mass ratio 1:10 (train range 0.25-4)."""
    return [
        {"x": 15.0, "y": 15.0, "vx": 2.0, "vy": 0.0, "radius": 0.75, "mass": 1.0},
        {"x": 20.0, "y": 15.3, "vx": 0.0, "vy": 0.0, "radius": 0.75, "mass": 10.0},
    ]


def scenario_unseen_wall_density(rng, n=40):
    """Held-out: wall segment sampled at spacing 4.0, above training's
    0.6-2.0 range -- tests robustness to sparser-than-trained sampling
    (spec §4 boundary-sampling question #3)."""
    spacing, point_radius = 4.0, ob.default_point_radius(4.0)
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
           for t in (5, 10, 20) if t <= steps}
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
    ap.add_argument("--gravity", type=float, default=9.0)
    ap.add_argument("--stiffness", type=float, default=400.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--neighbor-radius", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/contact_generalization_v1.json")
    args = ap.parse_args()

    dyn = ContactForceDynamics(dt=args.dt, neighbor_radius=args.neighbor_radius)
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
```

- [ ] **Step 2: Run on Polaris GPU against the Task 6 checkpoint**

Run (via Polaris job or interactive GPU session per [[feedback_generate_data_on_gpu]]): `PYTHONPATH=. python3 -m scripts.eval_contact_generalization --checkpoint checkpoints/contact_dynamics_v1.pt --device cuda`
Expected: exits 0; `kinematic_drift` is `0.0` (or within float32 noise, `< 1e-4`) for every scenario (obstacles never moved); `edges/node` in `tiling_scaling` stays within ~2x across the three `n_points` (O(N), not blowing up).

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_contact_generalization.py
git commit -m "Phase 1 held-out generalization eval (unseen obstacle shape/density, mass ratio, O(N) check)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

### Task 8: Rollout video + unbiased review

**Files:**
- Create: `scripts/render_contact_rollout.py`
- Reuse: pattern from `docs/debugging/frame-artifact-review-prompt.md` / `rollout-frame-review-prompt.md` per [[feedback_reuse_saved_subagent_prompts]] and [[feedback_auto_run_video_analysis_on_sim_finish]]

**Interfaces:**
- Consumes: `checkpoints/contact_dynamics_v1.pt`, `model.contact_force.ContactForceDynamics`, `model.obstacles`, `scripts.train_contact_dynamics.unroll`.
- Produces: `videos/contact_dynamics_v1_rollout.mp4` (per [[feedback_save_videos_to_videos_folder]]), delivered via SendUserFile per [[feedback_render_video_every_trained_model]] and [[feedback_subagents_cannot_send_files]].

- [ ] **Step 1: Write the rendering script**

```python
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
from scripts.eval_contact_generalization import scenario_unseen_obstacle_shape
from scripts.train_contact_dynamics import unroll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/contact_dynamics_v1.pt")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gravity", type=float, default=9.0)
    ap.add_argument("--stiffness", type=float, default=400.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--neighbor-radius", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="videos/contact_dynamics_v1_rollout.mp4")
    args = ap.parse_args()

    import random
    balls = scenario_unseen_obstacle_shape(random.Random(4738))
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
```

- [ ] **Step 2: Render the video, respecting the local-render memory-bounds rule** — confirm frame count (`args.steps`) x figure resolution stays well under RAM before running (this script already streams via `FFMpegWriter.grab_frame()` per frame rather than stacking, so no separate estimate is needed unless `--steps` is raised by an order of magnitude).

Run: `PYTHONPATH=. python3 -m scripts.render_contact_rollout --checkpoint checkpoints/contact_dynamics_v1.pt --device cuda --out videos/contact_dynamics_v1_rollout.mp4`
Expected: exits 0, `videos/contact_dynamics_v1_rollout.mp4` exists and plays.

- [ ] **Step 3: Dispatch a fresh, unbiased subagent** (no hypothesis primed) to describe what it sees in the rendered video, per [[feedback_auto_run_video_analysis_on_sim_finish]] — reuse `docs/debugging/rollout-frame-review-prompt.md`'s template per [[feedback_reuse_saved_subagent_prompts]] rather than writing a new prompt from scratch.

- [ ] **Step 4: Deliver the video via SendUserFile** and report the review findings plainly (per [[feedback_render_video_every_trained_model]]) — do not rely on aggregate err@k alone.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_contact_rollout.py
git commit -m "render + review Phase 1 contact-dynamics rollout video

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MQPJFSv1FcUmNosLqDbwrp"
```

---

## Notes for the executor

- Phase 2 (spin, tangential/friction force) is explicitly out of scope for every task above — do not add a spin channel, torque term, or non-conservative force anywhere in this plan (spec §0/§3/§9).
- The radius-0/no-obstacle parity claim (spec §5) is checked structurally by Task 4's `test_zero_init_is_free_flight` test, but the *full* empirical parity claim ("this reduces exactly to `gravity_central_v1.pt`'s behavior") is only checked once Task 6's checkpoint exists — if it's worth doing, add it as a follow-up eval script loading both checkpoints and comparing rollouts at `radius=0`, not blocking this plan's completion.
- Mass ratio range (train 0.25-4, held-out test 0.1/10) reuses the exact numbers already recommended in `docs/debugging/z-and-mass-channels-feasibility.md` §mass channel — don't re-derive a different range.
