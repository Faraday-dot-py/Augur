import torch

from scripts import hetero_ext_checks as hx
from scripts import scatter_tier0 as t0m
from scripts import train_scatter_field as tsf


def test_far_expansion_matches_direct_sum_inside_window():
    from scripts import scatter_scaling_ops as ops
    from scripts.scatter_scaling import make_scene
    p0, _ = make_scene(300, 14.0, 1.0, 128.0, 9000, 0)
    P = torch.tensor(p0)
    cc = torch.tensor([0.0, 0.0], dtype=torch.float64)
    a0, J, Q, C = hx.far_of(P, cc)
    inw = (P.abs() < hx.HALF).all(-1)
    core = (P.abs() < 16).all(-1)
    x = P[inw & core][None]
    from scripts.tier0_harvest import acc3
    approx = acc3(a0, J, Q, C, x)[0]
    full = ops.exact_acc(P, hx.EPS)[inw & core] - ops.exact_acc(P[inw], hx.EPS)[core[inw]]
    assert (approx - full).norm() / full.norm() < 0.5
    assert torch.isfinite(approx).all()
