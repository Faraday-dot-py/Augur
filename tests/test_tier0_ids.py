import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m


def _model():
    torch.manual_seed(4738)
    m = sf.ScatterField(dt=0.1, grid=32, extent=32.0, net="local", potential=True, verlet=True, width=8, layers=3)
    t0m.attach_tier0(m, "head=8,phid=4")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def test_hidden_keyed_by_id_survives_membership_change():
    m = _model()
    g = torch.Generator().manual_seed(1)
    pos = (torch.rand(1, 10, 2, generator=g) - 0.5) * 12
    vel = torch.randn(1, 10, 2, generator=g)
    one = torch.ones(1, 10)
    m.t0.set_ids(torch.arange(10)[None], 10)
    st = (m.init_field(1, pos.device)[0] if m.verlet else m.init_field(1, pos.device), None)
    for _ in range(2):
        pos, vel, st, _ = m.step(pos, vel, one, one, st)
    store = st[0][1]
    assert store.shape == (1, 10, 4) and float(store.abs().max()) > 0
    sel = torch.tensor([[7, 2, 5, 3]])
    m.t0.set_ids(sel, 10)
    p2, v2 = pos[0, sel[0]][None], vel[0, sel[0]][None]
    ctx_in = m.t0.begin(m, p2, v2, torch.ones(1, 4), torch.ones(1, 4), (st[0][0], store))[1]
    assert torch.allclose(ctx_in.ph[0], store[0, sel[0]])
    m.t0.set_ids(None, 0)
