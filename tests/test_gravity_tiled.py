import torch

from model.central_force import CentralForceDynamics
from scripts.gravity_1b import model_force, tiled_accel


def test_tiled_accel_matches_untiled():
    torch.manual_seed(4738)
    dyn = CentralForceDynamics(neighbor_radius=10.0)
    for p in dyn.parameters():
        torch.nn.init.normal_(p, std=0.3)
    pos = torch.rand(3000, 2) * 150
    full = dyn.accel(pos)
    tiled = tiled_accel(model_force(dyn), pos, 10.0, 7)
    assert torch.allclose(full, tiled, atol=1e-4, rtol=1e-4)
