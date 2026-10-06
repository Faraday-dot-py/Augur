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
