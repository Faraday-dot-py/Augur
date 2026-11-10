import torch

from model import local_field_medium as lfm
from scripts import finite_speed_local_field_medium as fs


def make(grid=32, seed=4738):
    torch.manual_seed(seed)
    m = lfm.LocalFieldMedium(grid=grid, extent=32.0, width=8)
    torch.nn.init.normal_(m.head.weight, std=0.3)
    torch.nn.init.normal_(m.head.bias, std=0.3)
    return m.eval()


def bodies(src_xy, probe_xy, src_mass=1.0):
    pos = torch.tensor([[src_xy, probe_xy]], dtype=torch.float32)
    mass = torch.tensor([[src_mass, 0.0]])
    return pos, mass, torch.ones(1, 2)


def test_zero_init_gives_zero_force():
    torch.manual_seed(4738)
    m = lfm.LocalFieldMedium(grid=32, extent=32.0, width=8).eval()
    pos, mass, mask = bodies([-5.0, 0.0], [5.0, 0.0])
    S = m.init_state(1, pos.device)
    with torch.no_grad():
        S, a = m.tick(S, pos, mass, mask, 2)
    assert float(a.abs().max()) == 0.0 and float(S.abs().max()) == 0.0


def test_signal_cannot_outrun_speed_bound():
    m = make()
    far_pos, mass, mask = bodies([-14.0, -14.0], [14.0, 14.0])
    S = m.init_state(1, far_pos.device)
    with torch.no_grad():
        _, a_src = m.tick(S, far_pos, mass, mask, 1)
        _, a_none = m.tick(S, far_pos, torch.zeros_like(mass), mask, 1)
    assert torch.equal(a_src[0, 1], a_none[0, 1])
    assert not torch.equal(a_src[0, 0], a_none[0, 0])


def test_more_substeps_reach_farther():
    m = make()
    pos, mass, mask = bodies([-4.0, 0.0], [4.0, 0.0])
    S = m.init_state(1, pos.device)
    with torch.no_grad():
        _, a1 = m.tick(S, pos, mass, mask, 1)
        _, a1z = m.tick(S, pos, torch.zeros_like(mass), mask, 1)
        _, a8 = m.tick(S, pos, mass, mask, 8)
        _, a8z = m.tick(S, pos, torch.zeros_like(mass), mask, 8)
    assert torch.equal(a1[0, 1], a1z[0, 1])
    assert not torch.equal(a8[0, 1], a8z[0, 1])


def test_retarded_equals_instantaneous_for_static_sources():
    P = torch.tensor([[-3.0, 0.0], [4.0, 1.0], [0.0, 5.0]]).repeat(6, 1, 1)
    R = fs.retarded_acc(P, 10.0)
    d = P[0][None] - P[0][:, None]
    inv = ((d ** 2).sum(-1) + fs.EPS ** 2) ** -1.5
    inv.fill_diagonal_(0.0)
    ref = (d * inv[..., None]).sum(1)
    assert torch.allclose(R[-1], ref, atol=1e-5)
    assert torch.allclose(R[0], ref, atol=1e-5)


def test_retarded_force_lags_a_moving_source():
    T = 40
    x = torch.linspace(-10.0, 10.0, T + 1)
    P = torch.stack([torch.stack([x, torch.zeros_like(x)], -1), torch.zeros(T + 1, 2).fill_(0.0) + torch.tensor([0.0, 6.0])], 1)
    slow = fs.retarded_acc(P, 5.0)
    fast = fs.retarded_acc(P, 5000.0)
    assert float((slow[T // 2:, 1, 0] - fast[T // 2:, 1, 0]).abs().max()) > 1e-3
