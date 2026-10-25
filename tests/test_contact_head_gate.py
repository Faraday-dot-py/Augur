import torch

from scripts import scatter_contact_scenes as cs


def test_gate_zero_in_free_flight_and_runs():
    torch.manual_seed(0)
    m = cs.build("B", "cpu", "none", "head=8,headmax=5,headgate,vclip=15")
    with torch.no_grad():
        for p in m.t0.headnet.parameters():
            p.add_(0.2 * torch.randn_like(p))
    m.eval()
    pos = torch.tensor([[[0.0, 0.0], [20.0, 20.0], [20.0, 21.0]]])
    vel = torch.ones(1, 3, 2)
    one = torch.ones(1, 3)
    g = m.contact_gate(pos, one)
    assert g[0, 0] == 0 and g[0, 1] > 0.9 and g[0, 2] > 0.9
    f = m.init_field(1, "cpu")[0]
    with torch.no_grad():
        a, _ = m.force(pos, vel, one, one, f)
    assert torch.isfinite(a).all() and float(m.t0.gate[0, 0]) == 0
