import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m


def _mk(spec):
    torch.manual_seed(4738)
    m = sf.ScatterField(dt=0.1, grid=32, extent=32.0, net="local", potential=True, verlet=True, width=8, layers=3)
    return t0m.attach_tier0(m, spec)


def test_matgate_zero_head_where_material_zero():
    m = _mk("head=8,mat=1,matgate")
    for p in m.t0.headnet.parameters():
        torch.nn.init.normal_(p, std=0.5)
    pos = (torch.rand(1, 12, 2, generator=torch.Generator().manual_seed(1)) - 0.5) * 10
    vel = torch.randn(1, 12, 2, generator=torch.Generator().manual_seed(2))
    one = torch.ones(1, 12)
    mat = torch.zeros(1, 12, 1)
    mat[0, :6] = 0.5
    m.t0.set_material(mat)
    m.t0.record = True
    with torch.no_grad():
        m.force(pos, vel, one, one, m.init_field(1, "cpu")[0])
    ha = m.t0.aux["head_acc"]
    assert ha[0, :6].abs().sum() > 0
    assert ha[0, 6:].abs().sum() == 0
