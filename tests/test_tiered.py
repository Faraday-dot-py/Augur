import pytest
import torch
import torch.nn as nn

from model.tiered import GatedRefine, TierNet, TieredField, TileLayout, ZeroTier0, deposit_moments, merge2x2, moment_features, nfeat, residual_penalty, seam_loss
from model.tiered.blend import smoother
from model.tiered.moments import M, NMOM, OCC
from model.tiered.sparse import smooth_gate

DT = torch.float64


def particles(B=2, N=300, extent=32.0, seed=4738):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(B, N, 2, generator=g, dtype=DT) - 0.5) * extent * 0.98
    vel = torch.randn(B, N, 2, generator=g, dtype=DT)
    mass = torch.rand(B, N, generator=g, dtype=DT) + 0.1
    mask = (torch.rand(B, N, generator=g, dtype=DT) > 0.1).to(DT)
    return pos, vel, mass, mask


def rand_coefs(layout, B, K=6, seed=4738, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, layout.T, layout.T, K, layout.P, layout.P, generator=g, dtype=DT) * scale


def global_coefs(layout, B, K=6, seed=4738):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, K, layout.n, layout.n, generator=g, dtype=DT)


def layout_for(periodic, blend=1.0, n=16, G=4, halo=3, h=1.0):
    return TileLayout(n, G, halo, blend, h, torch.full((2,), -n * h / 2), periodic)


def test_merge_exact_vs_direct_sums():
    pos, vel, mass, mask = particles()
    origin = torch.full((2,), -16.0, dtype=DT)
    occ = torch.rand(2, 1, 16, 16, dtype=DT)
    fine = deposit_moments(pos, vel, mass, mask, 16, 2.0, origin, occ=occ)
    mid = merge2x2(fine, 2.0)
    top = merge2x2(mid, 4.0)
    for mom, n, h in ((mid, 8, 4.0), (top, 4, 8.0)):
        direct = deposit_moments(pos, vel, mass, mask, n, h, origin)
        keep = [i for i in range(NMOM) if i != OCC]
        assert torch.allclose(mom[:, keep], direct[:, keep], atol=1e-10, rtol=1e-10)
    assert torch.allclose(mid[:, OCC], occ[:, 0].reshape(2, 8, 2, 8, 2).mean((2, 4)))
    assert fine.shape[1] == NMOM


def test_features_match_brute_force_cell_statistics():
    pos, vel, mass, mask = particles(B=1, N=400)
    origin = torch.full((2,), -16.0, dtype=DT)
    fine = deposit_moments(pos, vel, mass, mask, 16, 2.0, origin)
    top = merge2x2(merge2x2(merge2x2(merge2x2(fine, 2.0), 4.0), 8.0), 16.0)
    f = moment_features(top, 32.0, disp=True, cross=True)
    assert f.shape[1] == nfeat(True, True)
    w = (mass * mask)[0]
    m = w.sum()
    c = (w[:, None] * (pos[0] - origin - 16.0)).sum(0) / m
    d = pos[0] - origin - 16.0 - c
    q = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / m
    vbar = (w[:, None] * vel[0]).sum(0) / m
    dv = vel[0] - vbar
    wv = (w[:, None, None] * dv[:, :, None] * dv[:, None, :]).sum(0) / m
    x = (w[:, None, None] * d[:, :, None] * dv[:, None, :]).sum(0) / m
    got = f[0, :, 0, 0]
    assert torch.allclose(got[0], m / 32.0 ** 2)
    assert torch.allclose(got[1:3], vbar)
    assert torch.allclose(got[3:5], c / 32.0)
    assert torch.allclose(got[5:8], torch.stack([q[0, 0], q[0, 1], q[1, 1]]) / 32.0 ** 2)
    assert torch.allclose(got[8:11], torch.stack([wv[0, 0], wv[0, 1], wv[1, 1]]))
    assert torch.allclose(got[11:15], x.reshape(4) / 32.0)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("blend", [0.0, 1.0])
def test_partition_of_unity_sums_to_one(periodic, blend):
    L = layout_for(periodic, blend)
    pos = (torch.rand(2, 2000, 2, dtype=DT, generator=torch.Generator().manual_seed(4738)) - 0.5) * 16.0
    ones = torch.ones(2, L.T, L.T, 1, L.P, L.P, dtype=DT)
    v, g, _ = L.blend_eval(ones, pos)
    assert torch.allclose(v, torch.ones_like(v), atol=1e-12)
    assert g.abs().max() < 1e-9


@pytest.mark.parametrize("periodic", [False, True])
def test_blend_equals_cell_partition_reference_when_tiles_agree(periodic):
    L = layout_for(periodic)
    gc = global_coefs(L, 1)
    tiles = L.extract(gc)
    pos = (torch.rand(1, 60, 2, dtype=DT, generator=torch.Generator().manual_seed(1)) - 0.5) * 15.9
    v, g, _ = L.blend_eval(tiles, pos)
    pad = nn.functional.pad(gc, (2,) * 4, mode="circular" if periodic else "constant")
    for i in range(pos.shape[1]):
        s = (pos[0, i] - L.origin.to(DT)) / L.h
        val, grad = 0.0, torch.zeros(2, dtype=DT)
        c0 = torch.floor(s - 0.5)
        f = s - 0.5 - c0
        r, dr = smoother(f)
        for a in (0, 1):
            for b in (0, 1):
                wx, wy = (r[0] if a else 1 - r[0]), (r[1] if b else 1 - r[1])
                dwx, dwy = (dr[0] if a else -dr[0]) / L.h, (dr[1] if b else -dr[1]) / L.h
                ci, cj = int(c0[0]) + a, int(c0[1]) + b
                co = pad[0, :, cj + 2, ci + 2]
                dx, dy = (s[0] - ci - 0.5) * L.h, (s[1] - cj - 0.5) * L.h
                p = co[0] + co[1] * dx + co[2] * dy + 0.5 * co[3] * dx * dx + co[4] * dx * dy + 0.5 * co[5] * dy * dy
                gp = torch.stack([co[1] + co[3] * dx + co[4] * dy, co[2] + co[4] * dx + co[5] * dy])
                val = val + wx * wy * p
                grad = grad + torch.stack([dwx * wy * p, wx * dwy * p]) + wx * wy * gp
        assert abs(float(val) - float(v[0, i])) < 1e-9
        assert torch.allclose(grad, g[0, i], atol=1e-9)


@pytest.mark.parametrize("periodic", [False, True])
def test_force_is_true_gradient(periodic):
    L = layout_for(periodic)
    tiles = rand_coefs(L, 1, 8)
    pos = ((torch.rand(1, 500, 2, dtype=DT, generator=torch.Generator().manual_seed(2)) - 0.5) * 15.9).requires_grad_(True)
    v, g, _ = L.blend_eval(tiles, pos)
    auto = torch.autograd.grad(v.sum(), pos)[0]
    assert torch.allclose(auto, g, atol=1e-9)


def test_circulation_is_zero_across_seams():
    L = layout_for(False, 1.5, n=18, G=6, halo=4)
    tiles = rand_coefs(L, 1, 6)
    th = torch.linspace(0, 2 * torch.pi, 40001, dtype=DT)
    for cx, cy, r in ((0.0, 0.0, 3.0), (-2.0, 1.0, 2.5), (3.7, -3.3, 1.3)):
        loop = torch.stack([cx + r * torch.cos(th), cy + r * torch.sin(th)], -1)[None]
        _, g, _ = L.blend_eval(tiles, loop)
        tang = torch.stack([-torch.sin(th), torch.cos(th)], -1)[None] * r
        integrand = (g * tang).sum(-1)[0]
        circ = torch.trapezoid(integrand, th)
        assert abs(float(circ)) < 1e-5 * float(g.abs().max())


def test_residual_channel_is_not_curl_free():
    L = layout_for(False)
    tiles = torch.zeros(1, L.T, L.T, 8, L.P, L.P, dtype=DT)
    yy, xx = torch.meshgrid(torch.arange(L.P, dtype=DT), torch.arange(L.P, dtype=DT), indexing="ij")
    tiles[:, :, :, 6] = -yy
    tiles[:, :, :, 7] = xx
    act = torch.ones(1, L.T, L.T, dtype=torch.bool)
    mag, curl = residual_penalty(tiles, act, L.h)
    assert mag > 0 and curl > 1.0
    tiles[:, :, :, 6] = 1.0
    tiles[:, :, :, 7] = 0.0
    assert residual_penalty(tiles, act, L.h)[1] < 1e-12


@pytest.mark.parametrize("periodic", [False, True])
def test_seam_continuity_of_potential_and_force(periodic):
    L = layout_for(periodic, 1.0)
    tiles = rand_coefs(L, 1, 6)
    eps = 1e-7
    lo = -8.0 + 1e-3
    seams = [-8.0 + L.G * k for k in range(1, L.T)] + [-8.0 + L.G * k + 0.5 * L.G for k in range(L.T)] + [-8.0 + k + 0.5 for k in range(1, 15)]
    t = torch.linspace(lo, 8.0 - 1e-3, 37, dtype=DT)
    for s in seams:
        for axis in (0, 1):
            a = torch.stack([torch.full_like(t, s - eps), t], -1) if axis == 0 else torch.stack([t, torch.full_like(t, s - eps)], -1)
            b = a.clone()
            b[:, axis] = s + eps
            va, ga, _ = L.blend_eval(tiles, a[None])
            vb, gb, _ = L.blend_eval(tiles, b[None])
            assert (va - vb).abs().max() < 1e-5
            assert (ga - gb).abs().max() < 1e-4


def test_seam_loss_zero_for_consistent_tiles_and_positive_otherwise():
    for periodic in (False, True):
        L = layout_for(periodic)
        act = torch.ones(1, L.T, L.T, dtype=torch.bool)
        tiles = L.extract(global_coefs(L, 1, 6))
        assert seam_loss(tiles, act, L) < 1e-20
        assert seam_loss(rand_coefs(L, 1, 6), act, L) > 0.1


def make_nets(K, width=8, hidden=0, latent=0, residual=True, feedback=False, disp=True, cross=False, h0=1.0, seed=4738):
    torch.manual_seed(seed)
    nets, child = [], 0
    for k in range(1, K + 1):
        ext = 3 if feedback and k < K else 0
        net = TierNet(nfeat(disp, cross) + child + hidden + ext, h0 * 2 ** k, width, 2, residual, latent, True, hidden).to(DT)
        nn.init.normal_(net.out.weight, std=0.05)
        nn.init.normal_(net.out.bias, std=0.05)
        nets.append(net)
        child = net.summary_ch
    return nets


def make_field(K=2, periodic=True, hidden=0, latent=0, residual=True, feedback=False, cross=False, **kw):
    nets = make_nets(K, hidden=hidden, latent=latent, residual=residual, feedback=feedback, cross=cross)
    return TieredField(ZeroTier0(), nets, 32, 1.0, 4, 3, 1.0, periodic=periodic, feedback=feedback, cross=cross, **kw)


def test_translation_equivariance_periodic():
    f = make_field(2, True)
    pos, vel, mass, mask = particles()
    st = f.init_state(2, pos.device, DT)
    a, _, _ = f.force(pos, vel, mass, mask, st)
    shift = torch.tensor([16.0, -16.0], dtype=DT)
    p2 = (pos + 16.0 + shift) % 32.0 - 16.0
    b, _, _ = f.force(p2, vel, mass, mask, st)
    assert a.abs().max() > 1e-6
    assert torch.allclose(a, b, atol=1e-9)


def test_momentum_conserved_with_mean_removal():
    f = make_field(2, False)
    pos, vel, mass, mask = particles()
    pos0, vel0 = pos, vel
    st = f.init_state(2, pos.device, DT)
    a, _, _ = f.force(pos, vel, mass, mask, st)
    assert a.abs().max() > 1e-6
    assert ((mass * mask)[..., None] * a).sum(1).abs().max() < 1e-9
    p0 = ((mass * mask)[..., None] * vel).sum(1)
    for _ in range(3):
        pos, vel, st, _ = f.step(pos, vel, mass, mask, st)
    assert torch.allclose(((mass * mask)[..., None] * vel).sum(1), p0, atol=1e-9)
    g = make_field(2, False, mean_remove=False)
    b, _, _ = g.force(pos0, vel0, mass, mask, g.init_state(2, pos.device, DT))
    assert ((mass * mask)[..., None] * b).sum(1).abs().max() > 1e-6


def test_sparse_activation_matches_dense():
    pos, vel, mass, mask = particles(N=40, extent=8.0)
    pos = pos + torch.tensor([6.0, -5.0], dtype=DT)
    f1 = make_field(2, False, dilate=1)
    f2 = make_field(2, False, dilate=4)
    st = f1.init_state(2, pos.device, DT)
    a, _, act1 = f1.force(pos, vel, mass, mask, st)
    b, _, act2 = f2.force(pos, vel, mass, mask, st)
    assert torch.allclose(a, b, atol=1e-10)
    assert act1[0].sum() < act2[0].sum()


def test_zero_init_tier_gives_zero_force():
    torch.manual_seed(4738)
    nets = [TierNet(nfeat(), 2.0, 8, 2).to(DT), TierNet(nfeat() + 1, 4.0, 8, 2).to(DT)]
    f = TieredField(ZeroTier0(), nets, 32, 1.0, 4, 3, 1.0)
    pos, vel, mass, mask = particles()
    a, _, _ = f.force(pos, vel, mass, mask, f.init_state(2, pos.device, DT))
    assert a.abs().max() == 0


def test_hidden_latent_feedback_respa_run_and_stay_finite():
    f = make_field(2, True, hidden=2, latent=3, feedback=True, cross=True)
    pos, vel, mass, mask = particles()
    st = f.init_state(2, pos.device, DT)
    for _ in range(2):
        pos, vel, st, _ = f.step(pos, vel, mass, mask, st)
    assert torch.isfinite(pos).all() and torch.isfinite(vel).all()
    assert st[1][0].abs().max() > 0
    assert st[2][1] is not None
    p0 = ((mass * mask)[..., None] * vel).sum(1)
    pos, vel, st = f.step_respa(pos, vel, mass, mask, st, 2)
    assert torch.isfinite(pos).all()
    assert torch.allclose(((mass * mask)[..., None] * vel).sum(1), p0, atol=1e-9)


def test_gated_refine_force_is_gradient_of_gated_potential():
    torch.manual_seed(4738)
    net = TierNet(nfeat(), 0.5, 8, 2, residual=False).to(DT)
    nn.init.normal_(net.out.weight, std=0.05)
    nn.init.normal_(net.out.bias, std=0.05)
    r = GatedRefine(net, 32, 4, 3, 1.0, 0.5, torch.full((2,), -8.0), 0.5, 4.0)
    pos, vel, mass, mask = particles(N=200, extent=16.0)
    a, coefs, act = r(pos, vel, mass, mask)
    assert a.abs().max() > 1e-8
    L = r.layout
    mom = deposit_moments(pos, vel, mass, mask, L.n, L.h, L.origin.to(DT))
    gt = L.extract(smooth_gate(mom[:, M:M + 1] / L.h ** 2, 0.5, 4.0))
    p = pos.clone().requires_grad_(True)
    phi, _, _ = L.blend_eval(coefs, p)
    g, _, _ = L.blend_eval(gt, p)
    auto = -torch.autograd.grad((g * phi).sum(), p)[0]
    assert torch.allclose(auto, a, atol=1e-9)


def test_static_and_verlet_step_conserve_momentum():
    nets = make_nets(2)
    for n in nets:
        n.log_rho = True
    f = TieredField(ZeroTier0(), nets, 32, 1.0, 4, 3, 1.0, static=True)
    pos, vel, mass, mask = particles()
    st = f.init_state(2, pos.device, DT)
    p0 = ((mass * mask)[..., None] * vel).sum(1)
    a_prev = None
    for _ in range(3):
        pos, vel, st, a_prev = f.step_verlet(pos, vel, mass, mask, st, a_prev)
    assert torch.isfinite(pos).all()
    assert torch.allclose(((mass * mask)[..., None] * vel).sum(1), p0, atol=1e-9)


def test_third_order_features_match_brute_force():
    pos, vel, mass, mask = particles(B=1, N=400)
    origin = torch.full((2,), -16.0, dtype=DT)
    fine = deposit_moments(pos, vel, mass, mask, 16, 2.0, origin)
    top = merge2x2(merge2x2(merge2x2(merge2x2(fine, 2.0), 4.0), 8.0), 16.0)
    f = moment_features(top, 32.0, disp=False, cross=False, order3=True)
    assert f.shape[1] == nfeat(False, False, True)
    w = (mass * mask)[0]
    d = pos[0] - origin - 16.0
    c = (w[:, None] * d).sum(0) / w.sum()
    d = d - c
    t = [(w * d[:, 0] ** 3).sum(), (w * d[:, 0] ** 2 * d[:, 1]).sum(), (w * d[:, 0] * d[:, 1] ** 2).sum(), (w * d[:, 1] ** 3).sum()]
    want = torch.stack(t) / w.sum() / 32.0 ** 3
    assert torch.allclose(f[0, 8:12, 0, 0], want, atol=1e-12)


def test_learned_pool_latent_err_and_per_tier_accs():
    torch.manual_seed(4738)
    n1 = TierNet(nfeat(), 2.0, 8, 2, False, 3, True, 0, learned_pool=True).to(DT)
    n2 = TierNet(nfeat() + 4, 4.0, 8, 2, False, 3, True, 0).to(DT)
    for n in (n1, n2):
        nn.init.normal_(n.out.weight, std=0.05)
        nn.init.normal_(n.out.bias, std=0.05)
    f = TieredField(ZeroTier0(), [n1, n2], 32, 1.0, 4, 3, 1.0, static=True)
    pos, vel, mass, mask = particles()
    a, _ = f.upper_force(pos, vel, mass, mask, f.init_state(2, pos.device, DT))
    assert n1.pool is not None and len(f.last_accs) == 2
    assert torch.allclose(f.last_accs[0] + f.last_accs[1], a)
    assert f.last_errs[0].shape == (2, 1, 16, 16) and (f.last_errs[1] >= 0).all()
    a.sum().backward()
    assert n1.pool.weight.grad.abs().sum() > 0


def energy_field(seed=4738, moments=False):
    from model.tiered import TierNetE, TieredFieldE
    torch.manual_seed(seed)
    nets = [TierNetE(0.5 * 2 ** k, width=8, layers=3, moments=moments) for k in (1, 2)]
    for n in nets:
        nn.init.normal_(n.out.weight, std=0.05)
    return TieredFieldE(ZeroTier0(), nets, 64, 0.5, 4, 3).double()


@pytest.mark.parametrize("moments", [False, True])
def test_energy_tier_force_is_minus_grad_of_scalar(moments):
    f = energy_field(moments=moments)
    pos, vel, mass, mask = particles(B=1, N=40, extent=28.0)
    a, _ = f.upper_force(pos, vel, mass, mask, f.init_state(1, "cpu", DT))
    for i in (0, 7, 19):
        for ax in (0, 1):
            e = torch.zeros_like(pos)
            e[0, i, ax] = 1e-5
            up = sum(f.tier_energy(k, pos + e, mass, mask).sum() for k in (1, 2))
            dn = sum(f.tier_energy(k, pos - e, mass, mask).sum() for k in (1, 2))
            fd = -(up - dn) / 2e-5 / mass[0, i] * mask[0, i]
            assert abs(float(fd) - float(a[0, i, ax])) < 1e-5 * max(1.0, abs(float(fd)))


@pytest.mark.parametrize("moments", [False, True])
def test_energy_tier_no_work_on_closed_path(moments):
    f = energy_field(moments=moments)
    pos, vel, mass, mask = particles(B=1, N=40, extent=28.0)
    mask[0, 0] = 1.0
    th = torch.linspace(0, 2 * torch.pi, 801, dtype=DT)
    w = 0.0
    prev = None
    for t in th:
        p = pos.clone()
        p[0, 0] = torch.stack([3.0 + 2.0 * torch.cos(t), -1.0 + 2.0 * torch.sin(t)])
        a, _ = f.upper_force(p, vel, mass, mask, f.init_state(1, "cpu", DT))
        cur = (p[0, 0], mass[0, 0] * a[0, 0])
        if prev is not None:
            w += float(((cur[1] + prev[1]) / 2 * (cur[0] - prev[0])).sum())
        prev = cur
    assert abs(w) < 1e-4
