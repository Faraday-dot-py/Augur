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
