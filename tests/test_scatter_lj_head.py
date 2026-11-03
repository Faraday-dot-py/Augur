import torch

from scripts import scatter_lj as sl
from test_scatter_lj_energy import _model, _state


def test_head_zero_init_identity_then_velocity_dependent_and_momentum_free():
    m = _model().double()
    pt = sl.PeriodicTiles(m)
    pos, vel, mask, bx = _state(60, [14.0, 13.0])
    lay = sl.layout(bx[0])
    fld = pt.init_field(1, lay, "cpu", torch.float64)
    with torch.no_grad():
        a0, _ = pt.force(pos, vel, mask, bx, lay, fld)
    m2 = sl.build_model(energy=True, head=16).double()
    m2.load_state_dict({k: v for k, v in m.state_dict().items()}, strict=False)
    pt2 = sl.PeriodicTiles(m2)
    with torch.no_grad():
        a1, _ = pt2.force(pos, vel, mask, bx, lay, fld)
    assert torch.allclose(a0, a1)
    assert pt2.take_pen().shape == (2,)
    torch.nn.init.normal_(m2.t0.headnet[-1].weight, std=0.1)
    with torch.no_grad():
        a2, _ = pt2.force(pos, vel, mask, bx, lay, fld)
        a3, _ = pt2.force(pos, -3 * vel, mask, bx, lay, fld)
    assert not torch.allclose(a2, a3)
    assert float(a2.sum(1).abs().max()) < 1e-8
