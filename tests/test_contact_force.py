import torch

from model.contact_force import ContactForceDynamics
from model.token_graph import build_radius_graph_cells


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


def test_energy_conserved_bounded():
    dyn = _dyn()
    pos = torch.tensor([[10.0, 10.0], [10.5, 10.0]])
    vel = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    radius = torch.tensor([0.75, 0.75])
    mass = torch.tensor([1.0, 1.0])
    kinematic = torch.zeros(2, dtype=torch.bool)
    hidden = torch.zeros(2, 1)

    def total_energy(pos, vel):
        ke = 0.5 * (mass.unsqueeze(-1) * vel ** 2).sum()
        edges = build_radius_graph_cells(pos, dyn.neighbor_radius)
        src, dst = edges[0], edges[1]
        rel = pos[src] - pos[dst]
        d = torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1e-12)
        r_sum = (radius[src] + radius[dst]).unsqueeze(-1)
        feat = dyn.pair_features(d, r_sum, mass[src].unsqueeze(-1), mass[dst].unsqueeze(-1))
        v = dyn.potential(feat).sum() / 2  # each undirected pair appears as two directed edges
        return ke + v

    e0 = total_energy(pos, vel)
    energies = [e0]
    for _ in range(20):
        dp, dv, hidden = dyn(pos, vel, hidden, radius, mass, kinematic)
        pos, vel = pos + vel * dyn.dt + dp, vel + dv
        energies.append(total_energy(pos, vel))
    energies = torch.stack(energies)
    assert torch.isfinite(energies).all()
    assert (energies - e0).abs().max() < 0.5  # bounded oscillation, not a runaway drift


def test_kinematic_body_pinned_during_predictor_step():
    dyn = _dyn()
    pos = torch.tensor([[10.0, 10.0], [10.5, 10.0]])
    vel = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
    radius = torch.tensor([0.75, 1.0])
    mass = torch.tensor([1.0, 1.0])
    kinematic = torch.tensor([False, True])
    dt = dyn.dt
    kin = kinematic.unsqueeze(-1)
    a0 = dyn.accel(pos, radius, mass)
    dp_ref = torch.where(kin, torch.zeros_like(a0), 0.5 * dt * dt * a0)
    a1 = dyn.accel(pos + vel * dt + dp_ref, radius, mass)
    dv_ref = 0.5 * dt * (a0 + a1)
    dv_ref = torch.where(kin, torch.zeros_like(dv_ref), dv_ref)

    dp, dv, _ = dyn(pos, vel, torch.zeros(2, 1), radius, mass, kinematic)
    assert torch.allclose(dv[0], dv_ref[0], atol=1e-6)
