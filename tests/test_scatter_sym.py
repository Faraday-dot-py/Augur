import torch

from scripts import scatter_field as sf

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make(sym, seed=4738):
    torch.manual_seed(seed)
    m = sf.ScatterField(grid=32, extent=16.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True).to(DEV)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    m.sym = sym
    return m.eval()


def test_net_equivariant_and_off_is_identity():
    m = make("d4")
    x = torch.randn(2, 6, 32, 32, device=DEV)
    ref = m.net(x)
    m.sym = ""
    assert torch.equal(m.net_fwd(x), ref)
    for sym in ("flip", "c4", "d4"):
        m.sym = sym
        base = m.net_fwd(x)
        for op in m.SYM_OPS[sym]:
            xo = m._sym_op(x, op, [1, 3])
            assert torch.allclose(m.net_fwd(xo), m._sym_op(base, op, []), atol=1e-4)


def test_force_mirror_and_transpose():
    m = make("d4")
    g = torch.Generator().manual_seed(1)
    pos = (torch.rand(1, 40, 2, generator=g) * 12 - 6).to(DEV)
    vel = torch.randn(1, 40, 2, generator=g).to(DEV)
    mass = (torch.rand(1, 40, generator=g) + 0.5).to(DEV)
    mask = torch.ones(1, 40, device=DEV)
    f0 = m.init_field(1, DEV)[0]
    a, _ = m.force(pos, vel, mass, mask, f0)
    s = torch.tensor([-1.0, 1.0], device=DEV)
    a2, _ = m.force(pos * s, vel * s, mass, mask, f0)
    assert torch.allclose(a2, a * s, atol=1e-4)
    a3, _ = m.force(pos.flip(-1), vel.flip(-1), mass, mask, f0)
    assert torch.allclose(a3, a.flip(-1), atol=1e-4)
