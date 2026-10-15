import torch

from scripts import scatter_field as sf
from scripts.fewshot_adapter import MassGainAdapter, n_params


def tiny_model():
    torch.manual_seed(0)
    m = sf.ScatterField(grid=32, extent=64.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, levels=3, width=8)
    for p in m.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
        p.requires_grad_(False)
    return m.eval()


def scene():
    g = torch.Generator().manual_seed(1)
    pos = (torch.rand(2, 12, 2, generator=g) - 0.5) * 20
    vel = torch.randn(2, 12, 2, generator=g)
    mass = torch.ones(2, 12)
    mass[:, 0] = 30.0
    return pos, vel, mass, torch.ones(2, 12)


def roll(m, steps=5):
    pos, vel, mass, mask = scene()
    field = m.init_field(2, pos.device)
    for _ in range(steps):
        pos, vel, _, field, _ = sf._step_m(m, pos, vel, mass, mask, field)
    return pos, vel


def test_off_and_zero_init_bitwise():
    m = tiny_model()
    ref = roll(m)
    ad = MassGainAdapter()
    ad.on = False
    m.adapter = ad
    off = roll(m)
    ad.on = True
    on0 = roll(m)
    for a, b, c in zip(ref, off, on0):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_adapter_only_trainable_and_small():
    m = tiny_model()
    ad = MassGainAdapter()
    m.adapter = ad
    assert n_params(ad) <= 5000
    with torch.no_grad():
        ad.mlp[-1].weight.add_(0.1)
    pos, vel = roll(m, 3)
    (pos ** 2).sum().backward()
    assert ad.mlp[-1].weight.grad is not None and ad.mlp[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith("adapter."))


def test_gated_zero_on_equal_mass():
    m = tiny_model()
    pos, vel, mass, mask = scene()
    mass = torch.ones_like(mass)
    field = m.init_field(2, pos.device)
    ref = sf._step_m(m, pos, vel, mass, mask, field)[0]
    ad = MassGainAdapter(gated=True)
    with torch.no_grad():
        ad.mlp[-1].weight.add_(0.5)
    m.adapter = ad
    got = sf._step_m(m, pos, vel, mass, mask, m.init_field(2, pos.device))[0]
    assert torch.equal(ref, got)
