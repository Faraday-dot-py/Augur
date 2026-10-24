import torch

from scripts import scatter_field as sf
from scripts.scatter_tiling import TiledScatter

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make(nonet=False, seed=4738):
    torch.manual_seed(seed)
    m = sf.ScatterField(grid=64, extent=32.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True,
                        nonet=nonet).double().to(DEV)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def scene(n=200, sigma=6.0, seed=4738):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.randn(n, 2, generator=g) * sigma).clamp(-28, 28).double()
    vel = torch.randn(n, 2, generator=g).double()
    return pos.to(DEV), vel.to(DEV), torch.ones(n, dtype=torch.double, device=DEV)


def single_step(m, grid, extent, pos, vel, mass, steps=1):
    m.grid, m.extent, m.h = grid, extent, extent / grid
    n = pos.shape[0]
    st = m.init_field(1, pos.device, pos.dtype)
    p, v = pos[None], vel[None]
    for _ in range(steps):
        p, v, st, _ = m.step(p, v, mass[None], torch.ones(1, n, dtype=pos.dtype, device=DEV), st)
    return p[0], v[0], st


def test_one_tile_equals_model():
    m = make()
    pos, vel, mass = scene()
    ps, vs, _ = single_step(m, 64, 32.0, pos, vel, mass, 3)
    m.grid, m.extent, m.h = 64, 32.0, 0.5
    t = TiledScatter(m, 1, 32.0, 0.0)
    st = (t.init_field(pos.device, pos.dtype), None)
    p, v = pos, vel
    for _ in range(3):
        p, v, st = t.step(p, v, mass, st)
    assert torch.allclose(p, ps, atol=1e-9) and torch.allclose(v, vs, atol=1e-9)


def test_full_halo_matches_single_grid_without_net():
    m = make(nonet=True)
    pos, vel, mass = scene(sigma=7.0)
    ps, vs, _ = single_step(m, 128, 64.0, pos, vel, mass, 2)
    m.grid, m.extent, m.h = 64, 32.0, 0.5
    t = TiledScatter(m, 2, 32.0, 32.0)
    st = (t.init_field(pos.device, pos.dtype), None)
    p, v = pos, vel
    for _ in range(2):
        p, v, st = t.step(p, v, mass, st)
    assert torch.allclose(p, ps, atol=1e-8) and torch.allclose(v, vs, atol=1e-8)


def test_far_field_reduces_error_and_momentum():
    m = make(nonet=True)
    pos, vel, mass = scene(sigma=7.0)
    m.grid, m.extent, m.h = 128, 64.0, 0.5
    n = pos.shape[0]
    a_ref, _ = m.force(pos[None], vel[None], mass[None], torch.ones(1, n, dtype=torch.double, device=DEV), m.init_field(1, pos.device, pos.dtype)[0])
    a_ref = a_ref[0]
    m.grid, m.extent, m.h = 64, 32.0, 0.5
    errs = {}
    for far in (False, True):
        t = TiledScatter(m, 2, 32.0, 16.0, far=far, far_c=4)
        a, _ = t.force(pos, vel, mass, t.init_field(pos.device, pos.dtype))
        errs[far] = float((a - a_ref).norm(dim=-1).sum() / a_ref.norm(dim=-1).sum())
        assert float((mass[:, None] * a).sum(0).abs().max()) < 1e-9
    assert errs[True] < 0.5 * errs[False]


def test_every_body_owned_once():
    m = make(nonet=True)
    pos, vel, mass = scene(sigma=20.0)
    m.grid, m.extent, m.h = 64, 32.0, 0.5
    t = TiledScatter(m, 2, 32.0, 16.0)
    P, V, M, K, OWN, INS, idx = t._pack(pos, vel, mass, [0, 1, 2, 3])
    own = torch.cat([i[:int(OWN[j].sum())] for j, i in enumerate(idx)])
    assert sorted(own.tolist()) == list(range(pos.shape[0]))
