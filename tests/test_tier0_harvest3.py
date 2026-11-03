import torch

from scripts import tier0_harvest as th
from scripts import tiered_truth as tt


def test_third_order_expansion_converges():
    torch.manual_seed(0)
    T, M = 2, 30
    src = torch.randn(T, M, 2, dtype=torch.float64) * 5 + torch.tensor([60.0, 0.0], dtype=torch.float64)
    mass = torch.rand(T, M, dtype=torch.float64) + 0.5
    ok = torch.ones(T, M, dtype=torch.float64)
    cen = torch.zeros(T, 2, dtype=torch.float64)
    a0, J, Q, C = th.far_third(cen, src, mass, ok)
    x = (torch.rand(T, 5, 2, dtype=torch.float64) - 0.5) * 16
    exact = torch.zeros(T, 5, 2, dtype=torch.float64)
    for t in range(T):
        d = src[t][None] - x[t][:, None]
        f = tt._fkl(d, tt.EPS)[0]
        exact[t] = (mass[t][None, :, None] * f).sum(1)
    e2 = (th.acc3(a0, J, Q, torch.zeros_like(C), x) - exact).norm()
    e3 = (th.acc3(a0, J, Q, C, x) - exact).norm()
    assert e3 < 0.5 * e2
