import torch

from model.central_force import CentralForceDynamics


def test_momentum_conserved_by_construction():
    torch.manual_seed(4738)
    dyn = CentralForceDynamics()
    for p in dyn.parameters():
        torch.nn.init.normal_(p, std=0.3)
    pos = torch.rand(20, 2) * 30
    vel = torch.randn(20, 2)
    dp, dv, _ = dyn(pos, vel, torch.zeros(20, 1))
    assert dv.abs().max() > 0
    assert torch.allclose(dv.sum(0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(dp.sum(0), torch.zeros(2), atol=1e-4)


def test_zero_init_is_free_flight():
    dyn = CentralForceDynamics()
    pos, vel = torch.rand(5, 2) * 10, torch.randn(5, 2)
    dp, dv, _ = dyn(pos, vel, torch.zeros(5, 1))
    assert torch.all(dp == 0) and torch.all(dv == 0)
