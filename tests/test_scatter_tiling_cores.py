import torch

from scripts import scatter_field as sf
from scripts.scatter_tiling import TiledScatter
from scripts.scatter_tiling_cores import CoreTiled, lpt_assign, split_cores, uniform_cores

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make(nonet=False, seed=4738):
    torch.manual_seed(seed)
    m = sf.ScatterField(grid=64, extent=32.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True,
                        nonet=nonet).double().to(DEV)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def scene(n=200, sigma=6.0, seed=4738, lim=30.0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.randn(n, 2, generator=g) * sigma).clamp(-lim, lim).double()
    vel = torch.randn(n, 2, generator=g).double()
    return pos.to(DEV), vel.to(DEV), torch.ones(n, dtype=torch.double, device=DEV)


def test_adapt_pair_term_matches_stock_and_qmask():
    m = make()
    n = 700
    pos, _, mass = scene(n, sigma=3.0, lim=30.0)
    pos = pos[None]
    mass = (mass * (0.5 + torch.rand(n, dtype=torch.double, device=DEV)))[None]
    mk = torch.ones(1, n, dtype=torch.double, device=DEV)
    mk[0, ::7] = 0
    a0 = m.pp_acc(pos, mass, mk)
    m.opt.add("adapt")
    a1 = m.pp_acc(pos, mass, mk)
    assert torch.allclose(a0, a1, atol=1e-9)
    q = (torch.arange(n, device=DEV) % 3 == 0).double()[None] * mk
    a2 = m.pp_acc(pos, mass, mk, qmask=q)
    assert torch.allclose(a2 * q[..., None], a0 * q[..., None], atol=1e-9)
    assert float((a2 * (1 - q[..., None])).abs().max()) == 0.0


def test_adapt_dense_core_exact():
    m = make()
    n = 900
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(1, n, 2, generator=g).double().to(DEV) * 0.05
    pos[:, 600:] *= 100
    mass = torch.ones(1, n, dtype=torch.double, device=DEV)
    a0 = m.pp_acc(pos, mass, mass)
    m.opt.add("adapt")
    assert torch.allclose(m.pp_acc(pos, mass, mass), a0, atol=1e-8)


def test_uniform_cores_equal_tiled_scatter():
    m = make()
    pos, vel, mass = scene(300, sigma=10.0)
    t = TiledScatter(m, 2, 32.0, 16.0, ghosts=True, field_x=False, far=True)
    a0, _ = t.force(pos, vel, mass, t.init_field(pos.device, pos.dtype))
    c = CoreTiled(m, uniform_cores(2, 32.0), domain=64.0, far=True)
    a1, _ = c.force(pos, vel, mass, c.init_field(pos.device, pos.dtype))
    assert torch.allclose(a0, a1, atol=1e-9)


def test_split_cores_partition_ownership_and_momentum():
    m = make(nonet=True)
    pos, vel, mass = scene(400, sigma=5.0)
    cores = split_cores(uniform_cores(2, 32.0), pos, 60, dom=64.0)
    assert len(cores) > 4
    c = CoreTiled(m, cores, domain=64.0, far=True)
    P, V, M, K, OWN, INS, idx = c._pack(pos, vel, mass, list(range(c.T)))
    own = torch.cat([i[:int(OWN[j].sum())] for j, i in enumerate(idx)])
    assert sorted(own.tolist()) == list(range(pos.shape[0]))
    a, _ = c.force(pos, vel, mass, c.init_field(pos.device, pos.dtype))
    assert float((mass[:, None] * a).sum(0).abs().max()) < 1e-9


def test_split_cores_far_reduces_error():
    m = make(nonet=True)
    pos, vel, mass = scene(400, sigma=9.0)
    m.grid, m.extent, m.h = 128, 64.0, 0.5
    n = pos.shape[0]
    a_ref = m.force(pos[None], vel[None], mass[None], torch.ones(1, n, dtype=torch.double, device=DEV), m.init_field(1, pos.device, pos.dtype)[0])[0][0]
    cores = split_cores(uniform_cores(2, 32.0), pos, 80, dom=64.0)
    errs = {}
    for far in (False, True):
        c = CoreTiled(m, cores, domain=64.0, far=far)
        a, _ = c.force(pos, vel, mass, c.init_field(pos.device, pos.dtype))
        errs[far] = float((a - a_ref).norm(dim=-1).sum() / a_ref.norm(dim=-1).sum())
    assert errs[True] < 0.5 * errs[False]


def test_pair_pruning_is_exact():
    m = make(nonet=True)
    pos, vel, mass = scene(500, sigma=6.0)
    pos[:20] = 45.0 + 0.3 * torch.randn(20, 2, dtype=pos.dtype, device=DEV)
    cores = split_cores(uniform_cores(2, 32.0), pos, 80, dom=64.0)
    a = []
    for prune in (False, True):
        c = CoreTiled(m, cores, domain=64.0, far=True, prune_pair=prune)
        a.append(c.force(pos, vel, mass, c.init_field(pos.device, pos.dtype))[0])
        if prune:
            assert sum(c.stats["pair_cand"]) < sum(o + g for o, g in zip(c.stats["own"], c.stats["ghost"]))
    assert torch.allclose(a[0], a[1], atol=1e-9)


def test_retile_field_roundtrip():
    m = make(nonet=True)
    pos, _, _ = scene(400, sigma=5.0)
    cores = split_cores(uniform_cores(2, 32.0), pos, 60, dom=64.0)
    c = CoreTiled(m, cores, domain=64.0, far=False)
    f = torch.randn(c.T, m.cf, c.G, c.G, dtype=torch.double, device=DEV)
    g1 = c.assemble_fields(f)
    g2 = c.assemble_fields(c.slice_fields(g1))
    assert torch.allclose(g1, g2)
    f2 = c.retile(uniform_cores(2, 32.0), f)
    assert f2.shape[0] == 4


def test_lpt_assign_covers_all():
    bins = lpt_assign([5, 1, 3, 3, 2, 8], 3)
    assert sorted(sum(bins, [])) == list(range(6))
    assert max(sum([5, 1, 3, 3, 2, 8][k] for k in b) for b in bins) <= 9
