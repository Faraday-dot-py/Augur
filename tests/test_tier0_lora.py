import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m


def _mk(spec):
    torch.manual_seed(4738)
    m = sf.ScatterField(dt=0.1, grid=32, extent=32.0, net="unet", potential=True, verlet=True, width=8, levels=2) if False else sf.ScatterField(dt=0.1, grid=32, extent=32.0, net="local", potential=True, verlet=True, width=8, layers=3)
    return t0m.attach_tier0(m, spec)


def test_lora_identity_at_init_and_trainable():
    a, b = _mk(""), _mk("lora=2")
    sd = {k: v for k, v in a.state_dict().items()}
    sf.load_compat(b, sd)
    pos = (torch.rand(1, 12, 2, generator=torch.Generator().manual_seed(1)) - 0.5) * 10
    vel = torch.randn(1, 12, 2, generator=torch.Generator().manual_seed(2))
    one = torch.ones(1, 12)
    with torch.no_grad():
        fa = a.init_field(1, "cpu")[0]
        fb = b.init_field(1, "cpu")[0]
        aa, _ = a.force(pos, vel, one, one, fa)
        ab, _ = b.force(pos, vel, one, one, fb)
    assert torch.allclose(aa, ab)
    assert any("lora_a" in n for n, _ in b.named_parameters())
    assert b.t0_spec == "lora=2"
