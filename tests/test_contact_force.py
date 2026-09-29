import torch

from model.contact_force import ContactForceDynamics


def _dyn():
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
