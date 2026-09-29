"""GPU-vectorized torch port of bounce.py's penalty-force physics (per-ball
radius/mass/kinematic, plus wall-segment obstacles), for generating Phase 1
contact-generalization training data at scale. Must stay bit-comparable
(within float precision) with bounce.py's CPU reference -- see
tests/test_contact_truth.py -- since bounce.py is the one source of truth
(spec docs/superpowers/specs/2026-09-28-cfd-contact-generalization-design.md §6/§9).

Usage: PYTHONPATH=. python3 -c "from scripts import contact_truth" (library module, no CLI)
"""
import numpy as np
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


def forces(p, radii, masses, n, gravity, stiffness, segments=None):
    """Net force (not yet divided by mass) on each body: gravity + walls +
    obstacle segments + pairwise ball contact. `p` is (m, 2), `radii` and
    `masses` (m,)."""
    xs, ys = p[:, 0], p[:, 1]
    wfx, wfy = _wall_force(xs, ys, n, radii, stiffness)
    base_x = gravity * masses + wfx
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
    p = torch.tensor([[b["x"], b["y"]] for b in balls], dtype=torch.float64, device=device)
    v = torch.tensor([[b["vx"], b["vy"]] for b in balls], dtype=torch.float64, device=device)
    radii = torch.tensor([b.get("radius", 0.75) for b in balls], dtype=torch.float64, device=device)
    masses = torch.tensor([b.get("mass", 1.0) for b in balls], dtype=torch.float64, device=device)
    kinematic = torch.tensor([b.get("kinematic", False) for b in balls], dtype=torch.bool, device=device)
    sub_dt = dt / substeps
    ps, vs = [p.cpu().numpy().copy()], [v.cpu().numpy().copy()]
    for _ in range(steps):
        for _ in range(substeps):
            f = forces(p, radii, masses, n, gravity, stiffness, segments)
            new_v = v + (f / masses.unsqueeze(-1)) * sub_dt
            v = torch.where(kinematic.unsqueeze(-1), v, new_v)
            new_p = p + v * sub_dt
            p = torch.where(kinematic.unsqueeze(-1), p, new_p)
        ps.append(p.cpu().numpy().copy())
        vs.append(v.cpu().numpy().copy())
    return np.array(ps), np.array(vs)
