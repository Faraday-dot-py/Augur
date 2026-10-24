import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import tiered_eval as te
from scripts import tiered_iface as ti
from scripts import tiered_scenes as sc
from scripts import tiered_truth as tt

CPU = torch.device("cpu")


def _rand(n, seed=4738, spread=10.0):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(1, n, 2, generator=g, dtype=torch.float64) - 0.5) * 2 * spread
    vel = torch.randn(1, n, 2, generator=g, dtype=torch.float64) * 0.3
    mass = torch.rand(1, n, generator=g, dtype=torch.float64) + 0.2
    return pos, vel, mass, torch.ones(1, n, dtype=torch.float64)


def test_far_expansion_matches_autograd():
    src = torch.randn(30, 2, dtype=torch.float64) * 4 + torch.tensor([25.0, -10.0], dtype=torch.float64)
    sm = torch.rand(30, dtype=torch.float64) + 0.5

    def acc(x):
        d = src - x
        return (d * ((d ** 2).sum(-1, keepdim=True) + tt.EPS ** 2) ** -1.5 * sm[:, None]).sum(0)

    x0 = torch.zeros(2, dtype=torch.float64)
    a0, J, Q = tt.far_expansion(x0[None], src, sm, torch.ones(1, 30, dtype=torch.float64))
    assert torch.allclose(a0[0], acc(x0), atol=1e-12)
    Jt = torch.autograd.functional.jacobian(acc, x0)
    assert torch.allclose(J[0], Jt, atol=1e-10)
    Ht = torch.autograd.functional.jacobian(lambda x: torch.autograd.functional.jacobian(acc, x, create_graph=True), x0)
    assert torch.allclose(Q[0], Ht, atol=1e-9)
    x = torch.tensor([1.5, -2.0], dtype=torch.float64)
    ex = tt.ExtField(a0, J, Q).acc(x[None, None])[0, 0]
    assert (ex - acc(x)).norm() / acc(x).norm() < 1e-2


def test_cell_gravity_close_to_direct():
    rng = np.random.default_rng(4738)
    pos, _, mass = sc.big_scene("clumpy", 3000, 128.0, rng, CPU)
    p, m = torch.as_tensor(pos), torch.as_tensor(mass)
    d = tt.grav_acc(p[None], m[None], torch.ones(1, 3000, dtype=torch.float64))[0]
    c = tt.cell_grav_acc(p, m)
    assert ((c - d) ** 2).sum().sqrt() / (d ** 2).sum().sqrt() < 0.02


def test_laws_are_gradients_of_their_energy():
    pos, vel, mass, mask = _rand(12, spread=2.5)
    for kw in (dict(grav=dict(eps=0.5)), dict(lj=dict(rc=2.5)), dict(contact=dict(radius=0.75, k=400.0))):
        ph = tt.Physics(**kw)
        p = pos.clone().requires_grad_(True)
        e = ph.pe(p, vel, mass, mask).sum()
        g, = torch.autograd.grad(e, p)
        a = ph.acc(pos, vel, mass, mask)
        assert torch.allclose(-g, a * mass[..., None], atol=1e-6, rtol=1e-6), kw
    ph = tt.Physics(walls=dict(lo=torch.full((1, 2), -3.0, dtype=torch.float64), hi=torch.full((1, 2), 3.0, dtype=torch.float64), radius=0.75, k=400.0))
    p = (pos * 1.3).clone().requires_grad_(True)
    g, = torch.autograd.grad(ph.pe(p, vel, mass, mask).sum(), p)
    assert torch.allclose(-g, ph.acc(p.detach(), vel, mass, mask) * mass[..., None], atol=1e-6)


def test_ext_field_curl_free_and_gradient():
    gen = torch.Generator().manual_seed(4738)
    e = tt.ExtField.random(3, gen, CPU, a_scale=0.3, j_scale=0.1, q_scale=0.01)
    assert torch.allclose(e.J, e.J.transpose(1, 2))
    pos, vel, mass, mask = _rand(8)
    pos = pos.expand(3, -1, -1).clone().requires_grad_(True)
    m, k = mass.expand(3, -1), mask.expand(3, -1)
    g, = torch.autograd.grad(e.pe(pos, m, k).sum(), pos)
    assert torch.allclose(-g, e.acc(pos.detach()) * m[..., None], atol=1e-9)
    ch = e.channels(8, 16.0)
    assert ch.shape == (3, 6, 8, 8)


def test_rollout_conserves_energy_momentum():
    pos, vel, mass, mask = _rand(20, spread=6.0)
    ph = tt.Physics(grav=dict(eps=0.5))
    r = tt.rollout(ph, pos, vel, mass, mask, 20, 0.1, 4)
    e0, e1 = ph.energy(r["pos"][:, 0], r["vel"][:, 0], mass, mask), ph.energy(r["pos"][:, -1], r["vel"][:, -1], mass, mask)
    assert abs(float(e1 - e0)) < 1e-3 * abs(float(e0)) + 1e-4
    p0, p1 = (mass[..., None] * r["vel"][:, 0]).sum(1), (mass[..., None] * r["vel"][:, -1]).sum(1)
    assert torch.allclose(p0, p1, atol=1e-9)


def test_periodic_lj_conserves_momentum_and_wraps():
    rng = np.random.default_rng(4738)
    a = sc.lj_spacing(0.7)
    pts = sc.hex_lattice(8, 9, a)
    box = np.array([8 * a, 9 * a * np.sqrt(3) / 2])
    n = len(pts)
    s = sc._empty("lj_liquid", 0.1, 20)
    s.update(pos=pts + rng.normal(0, 0.05 * a, pts.shape), vel=sc.maxwell(n, 0.6, rng), mass=np.ones(n), kmat=np.ones(n), law=dict(lj=dict(rc=2.5)),
             bound=dict(kind="periodic", period=box))
    b = sc.build_batch([s], CPU)
    r = tt.rollout(b["phys"], b["pos"], b["vel"], b["mass"], b["mask"], 5, 0.1, 20)
    assert torch.allclose(r["vel"][:, -1].sum(1), r["vel"][:, 0].sum(1), atol=1e-8)


def test_every_tier0_kind_builds_and_rolls():
    for kind in sc.TIER0_KINDS:
        rng = np.random.default_rng(4738)
        scenes = [sc.make_scene(kind, rng, 64) for _ in range(3)]
        keys = {}
        for s in scenes:
            keys.setdefault((tuple(sorted(s["law"])), s["bound"]["kind"], s["dt"], s["sub"]), []).append(s)
        for g in keys.values():
            b = sc.build_batch(g, CPU)
            assert (b["pos"].abs() <= sc.WINDOW).all(), kind
            ext = tt.ExtField.random(len(g), torch.Generator().manual_seed(4738), CPU)
            r = tt.rollout(b["phys"], b["pos"], b["vel"], b["mass"], b["mask"], 2, b["dt"], b["sub"], ext=ext)
            assert torch.isfinite(r["pos"]).all() and torch.isfinite(r["acc_near"]).all(), kind
            assert b["occ"].shape == (len(g), 3, sc.GC, sc.GC)
            total = r["acc_near"] + r["acc_ext"]
            assert torch.allclose(total - r["acc_ext"], r["acc_near"])


def test_boundary_occupancy_channels():
    rng = np.random.default_rng(4738)
    w = sc.occupancy(sc.boundary_scene(rng, 32, bkind="wall", base="grav")["bound"])
    o = sc.occupancy(sc.boundary_scene(rng, 32, bkind="open", base="grav")["bound"])
    p = sc.occupancy(sc.boundary_scene(rng, 32, bkind="periodic", base="grav")["bound"])
    assert w[0].max() > 0 and w[1].max() == 0 and o[1].max() > 0 and o[0].max() == 0 and p[2].min() == 1


def test_oracle_matches_truth_exactly():
    b = te.grav_batch("plummer", 30, 20.0, 4738, 2, CPU)
    out = te.compare(ti.Oracle(0.1), b, 5, ks=(1, 5))
    assert out["err"]["5"] < 1e-9 and out["drift"]["dE"]["5"]["model_minus_truth"] < 1e-9


def test_harness_ext_kick_matches_truth_for_constvel_zero_field():
    b = te.grav_batch("uniform", 10, 20.0, 4738, 1, CPU)
    b["phys"] = tt.Physics()
    ext = tt.ExtField(torch.tensor([[0.2, -0.1]], dtype=torch.float64), torch.zeros(1, 2, 2, dtype=torch.float64), torch.zeros(1, 2, 2, 2, dtype=torch.float64))
    out = te.compare(ti.ConstVel(0.1), b, 10, ks=(10,), ext=ext)
    assert out["err"]["10"] < 1e-9


def _tiny_scatter():
    torch.manual_seed(4738)
    m = sf.ScatterField(grid=32, extent=64.0, net="local", recurrent=True, hidden_ch=0, dt=0.1, width=8, layers=2, momfix=True)
    for p in m.net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return ti.ScatterAdapter(m.double(), "tiny", dtype=torch.float64)


def test_scatter_adapter_and_tiled_sparse_equals_dense():
    ad = _tiny_scatter()
    pos, vel, mass, mask = _rand(60, spread=20.0)
    p2, v2, st = ad.step(pos, vel, mass, mask, ad.init_state(1, CPU), None)
    assert torch.isfinite(p2).all()
    span = 128.0
    pos = pos * 3.0
    d = ti.Tiled(ad, span, sparse=False)
    s = ti.Tiled(ad, span, sparse=True)
    ad_, _ = d.force(pos, vel, mass, mask, d.init_state(1, CPU))
    as_, _ = s.force(pos, vel, mass, mask, s.init_state(1, CPU))
    assert torch.allclose(ad_, as_, atol=1e-10)
    assert s.last_active <= s.last_tiles


class _Truncated(ti.TierModel):
    name, reach, laws = "trunc", 64.0, frozenset({"grav"})

    def __init__(self, rc):
        self.rc = rc
        self.lower = ti.RangeLower(rc)

    def force(self, pos, vel, mass, mask, state, ctx=None):
        return self.lower(pos, vel, mass, mask), state


def test_tiled_matches_direct_when_halo_covers_force_range():
    pos, vel, mass, mask = _rand(80, spread=60.0)
    base = _Truncated(8.0)
    t = ti.Tiled(base, 128.0, sparse=True)
    a, _ = t.force(pos, vel, mass, mask, t.init_state(1, CPU))
    assert torch.allclose(a, base.lower(pos, vel, mass, mask), atol=1e-10)


def test_calibration_group_skipped_without_error_method():
    assert te.g_calibration(ti.ConstVel(), CPU, 4738, True).startswith("skipped")


def test_seam_probe_zero_for_exact_force_model():
    r = te.g_seams(ti.Oracle(0.1), CPU, 4738, True)
    assert r["spurious_work_across_seam"] < 1e-9
    assert abs(r["seam_jump_ratio_model"] - r["seam_jump_ratio_truth"]) < 1e-6
