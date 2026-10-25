import torch

from scripts import scatter_field as sf


def _model(**kw):
    torch.manual_seed(0)
    m = sf.ScatterField(grid=32, extent=16.0, net="unet", recurrent=True, hidden_ch=0, dt=0.1, momfix=True, kernel=True, pp=2.0, split=True,
                        potential=True, verlet=True, **kw)
    for mod in (m.net.out, m.kmlp[-1], m.ppmlp[-1]):
        torch.nn.init.normal_(mod.weight, std=0.05)
    return m


def test_energy_force_is_gradient_and_velocity_blind():
    m = _model(energy=True)
    pos = (torch.rand(1, 12, 2) - 0.5) * 8
    mass = torch.ones(1, 12)
    mask = torch.ones(1, 12)
    f0 = m.init_field(1, "cpu")[0]
    a1, _ = m.force(pos, torch.randn(1, 12, 2), mass, mask, f0)
    a2, _ = m.force(pos, torch.zeros(1, 12, 2), mass, mask, f0 + 1.0)
    assert torch.allclose(a1, a2, atol=1e-6)
    assert float(a1.abs().sum()) > 0
    assert float((a1 * mask[..., None]).sum(1).abs().max()) < 1e-5


def test_energy_forces_train_and_nongradient_flags():
    m = _model(energy=True)
    pos = ((torch.rand(2, 10, 2) - 0.5) * 8).requires_grad_(True)
    vel = torch.randn(2, 10, 2)
    mass, mask = torch.ones(2, 10), torch.ones(2, 10)
    f = m.init_field(2, "cpu")
    p2, v2, mass, f, _ = sf._step_m(m, pos, vel, mass, mask, f)
    ((p2 ** 2).sum() + (v2 ** 3).sum()).backward()
    assert float(m.net.out.weight.grad.abs().sum()) > 0
    ng = _model(vblind=True, norec=True)
    pos = (torch.rand(1, 12, 2) - 0.5) * 8
    mass, mask = torch.ones(1, 12), torch.ones(1, 12)
    f0 = ng.init_field(1, "cpu")[0]
    a1, _ = ng.force(pos, torch.randn(1, 12, 2), mass, mask, f0)
    a2, _ = ng.force(pos, torch.zeros(1, 12, 2), mass, mask, f0 + 1.0)
    assert torch.allclose(a1, a2, atol=1e-6)
