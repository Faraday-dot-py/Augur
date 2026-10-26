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


def test_gravity_respa_pair_sub1_matches_verlet_and_nsub_runs():
    m = _model(vblind=True, norec=True)
    pos = (torch.rand(1, 14, 2) - 0.5) * 8
    vel = torch.randn(1, 14, 2) * 0.1
    mass, mask = torch.ones(1, 14), torch.ones(1, 14)
    st = m.init_field(1, "cpu")
    with torch.no_grad():
        p1, v1, _, _ = m.step(pos, vel, mass, mask, st)
        m.pair_sub = 1
        p2, v2, _, _ = m.step_respa(pos, vel, mass, mask, st)
        assert torch.allclose(p1, p2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)
        m.pair_sub = 4
        p3, v3, _, dv = m.step(pos, vel, mass, mask, st)
        assert torch.isfinite(p3).all() and torch.allclose(dv, v3 - vel)
        m.pair_sub, m.nsub = 1, 2
        p4, v4, _, _ = m.step(pos, vel, mass, mask, st)
        assert torch.isfinite(p4).all() and m.dt == 0.1
