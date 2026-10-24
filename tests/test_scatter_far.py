import torch

from scripts import scatter_far as sfar
from scripts import scatter_field as sf
from scripts.scatter_tiling import TiledScatter

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make(seed=4738):
    torch.manual_seed(seed)
    m = sf.ScatterField(grid=64, extent=32.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True).to(DEV)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def test_tail_kernel_interior_and_continuity():
    m = make()
    tk = sfar.TailKernel(pp=m.pp, split=m.split, r0=32.0).to(DEV)
    tk.old.load_state_dict(m.kmlp.state_dict())
    r = torch.linspace(0.05, 32.0, 400, device=DEV)
    assert torch.equal(tk(r), sfar.old_fn(m)(r))
    eps = torch.tensor([31.999, 32.001], device=DEV)
    k = tk(eps)
    assert abs(float(k[1] - k[0])) < 1e-3
    far = torch.logspace(1.6, 3.2, 300, device=DEV)
    F = sfar.force_of(tk, far)
    assert bool((F > 0).all())


def test_install_restores_and_matches_old_fn():
    m = make()
    rho = torch.rand(1, 1, 64, 64, device=DEV)
    ref = m.kernel_acc(rho)
    sfar.install(m, sfar.old_fn(m))
    assert torch.allclose(m.kernel_acc(rho), ref, atol=1e-5)
    sfar.install(m, None)
    assert torch.equal(m.kernel_acc(rho), ref)


def test_cache_matches_uncached():
    m = make()
    rho = torch.rand(1, 1, 64, 64, device=DEV)
    sfar.install_spec(m, "oldf")
    a = m.kernel_acc(rho)
    sfar.install_spec(m, "oldf+cache")
    with torch.no_grad():
        b1 = m.kernel_acc(rho)
        b2 = m.kernel_acc(rho)
    assert torch.allclose(a, b1, atol=1e-6) and torch.equal(b1, b2)
    sfar.install(m, None)


def test_clamp_and_oracle_tails():
    m = make()
    r = torch.tensor([50.0, 100.0, 300.0], device=DEV)
    f40 = sfar.force_of(sfar.old_fn(m), torch.tensor([40.0], device=DEV))
    assert torch.allclose(sfar.force_of(sfar.clamp_fn(m, 40.0, "inv2"), r), f40 * (40.0 / r) ** 2, rtol=1e-3)
    assert bool((sfar.force_of(sfar.clamp_fn(m, 40.0, "zero"), r) == 0).all())
    assert torch.allclose(sfar.force_of(sfar.oracle_fn(m, 64.0), r + 50), sfar.f_true(r + 50), rtol=1e-3)


def test_tiled_far_runs_with_replacement_kernel():
    m = make()
    tk = sfar.TailKernel(pp=m.pp, split=m.split, r0=32.0).to(DEV)
    tk.old.load_state_dict(m.kmlp.state_dict())
    sfar.install(m, tk)
    T = TiledScatter(m, 2, 32, 16, ghosts=True, field_x=True, far=True)
    g = torch.Generator().manual_seed(4738)
    pos = (torch.randn(120, 2, generator=g) * 8).clamp(-28, 28).to(DEV)
    vel = torch.randn(120, 2, generator=g).to(DEV)
    mass = torch.ones(120, device=DEV)
    with torch.no_grad():
        a, _ = T.force(pos, vel, mass, T.init_field(DEV))
    assert torch.isfinite(a).all()
    sfar.install(m, None)
