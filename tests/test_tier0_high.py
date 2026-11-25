import torch

from scripts import scatter_field as sf
from scripts import scatter_tier0 as t0m


def test_ext_high_order_matches_einsum():
    m = sf.ScatterField(dt=0.1, grid=32, extent=32.0, net="local", potential=True, verlet=True, width=8, layers=3)
    t0m.attach_tier0(m, "ext=6,extapply")
    B, N = 3, 7
    g = torch.Generator().manual_seed(2)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    pos, Q, C = r(B, N, 2), r(B, 2, 2, 2), r(B, 2, 2, 2, 2)
    m.t0.set_external(m, phi0=torch.zeros(B, dtype=torch.float64), g=r(B, 2), H=r(B, 3), high=(Q, C))
    got = m.t0.ext_acc(m, pos)
    m.t0.ext_high = None
    base = m.t0.ext_acc(m, pos)
    ref = 0.5 * torch.einsum("bijk,bnj,bnk->bni", Q, pos, pos) + torch.einsum("bijkl,bnj,bnk,bnl->bni", C, pos, pos, pos) / 6
    assert torch.allclose(got - base, ref)
