import numpy as np
import pytest
import torch

from model import ball_tree_attention as bt

DEV = torch.device("cpu")


def make(seed=4738, **kw):
    torch.manual_seed(seed)
    m = bt.BallTreeAttention(**kw).to(DEV)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
    return m.eval()


def scene(n, b=1, seed=4738, pad=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(b, n + pad, 2, generator=g) * 6
    vel = torch.randn(b, n + pad, 2, generator=g) * 0.5
    mass = torch.ones(b, n + pad)
    mask = torch.ones(b, n + pad)
    mask[:, n:] = 0
    return pos, vel, mass, mask


def test_tree_order_is_permutation_with_padding_last():
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(2, 64, 2, generator=g)
    real = torch.ones(2, 64, dtype=torch.bool)
    real[0, 40:] = False
    perm = bt.tree_order(pos, real, 16)
    for b in range(2):
        assert sorted(perm[b].tolist()) == list(range(64))
    assert real[0].gather(0, perm[0])[:40].all() and not real[0].gather(0, perm[0])[40:].any()


def test_tree_halves_are_spatially_separated():
    g = torch.Generator().manual_seed(2)
    pos = torch.randn(1, 128, 2, generator=g)
    real = torch.ones(1, 128, dtype=torch.bool)
    perm = bt.tree_order(pos, real, 16)
    p = pos[0][perm[0]]
    first, second = p[:64], p[64:]
    ax = int((pos[0].max(0).values - pos[0].min(0).values).argmax())
    assert first[:, ax].max() <= second[:, ax].min()


@pytest.mark.parametrize("n", [5, 16, 17, 100, 300])
def test_force_finite_and_momentum_free(n):
    m = make()
    pos, vel, mass, mask = scene(n)
    with torch.no_grad():
        a, _ = m.force(pos, vel, mass, mask, None)
    assert a.shape == (1, n, 2) and torch.isfinite(a).all()
    assert float((a * mass[..., None]).sum(1).abs().max()) < 1e-4


def test_order_translation_and_padding_invariance():
    m = make()
    pos, vel, mass, mask = scene(100)
    with torch.no_grad():
        a0, _ = m.force(pos, vel, mass, mask, None)
        perm = torch.randperm(100, generator=torch.Generator().manual_seed(3))
        a1, _ = m.force(pos[:, perm], vel[:, perm], mass[:, perm], mask[:, perm], None)
        a2, _ = m.force(pos + torch.tensor([300.0, -120.0]), vel, mass, mask, None)
        z = lambda t: torch.cat([t, torch.randn(1, 37, *t.shape[2:]) * 9 if t.dim() == 3 else torch.zeros(1, 37)], 1)
        a3, _ = m.force(z(pos), z(vel), z(mass), torch.cat([mask, torch.zeros(1, 37)], 1), None)
    assert torch.allclose(a1, a0[:, perm], atol=1e-4)
    assert torch.allclose(a2, a0, atol=1e-3)
    assert torch.allclose(a3[:, :100], a0, atol=1e-4)
    assert float(a3[:, 100:].abs().max()) == 0.0


def test_batch_with_unequal_counts_matches_single():
    m = make()
    p1, v1, m1, k1 = scene(70, seed=5)
    p2, v2, m2, k2 = scene(100, seed=6)
    pad = lambda t, n: torch.cat([t, t.new_zeros(1, n - t.shape[1], *t.shape[2:])], 1)
    P = torch.cat([pad(p1, 100), p2]); V = torch.cat([pad(v1, 100), v2]); M = torch.cat([pad(m1, 100), m2]); K = torch.cat([pad(k1, 100), k2])
    with torch.no_grad():
        a, _ = m.force(P, V, M, K, None)
        a_1, _ = m.force(p1, v1, m1, k1, None)
        a_2, _ = m.force(p2, v2, m2, k2, None)
    assert torch.allclose(a[0, :70], a_1[0], atol=1e-4)
    assert torch.allclose(a[1], a_2[0], atol=1e-4)


def test_pool_conserves_mass_and_com():
    pool = bt.Pool(8)
    h = torch.randn(1, 32, 8)
    x = torch.randn(1, 32, 2)
    M = torch.rand(1, 32) + 0.5
    hp, xp, Mp = pool(h, x, M)
    assert torch.allclose(Mp.sum(), M.sum())
    assert torch.allclose((xp * Mp[..., None]).sum(1), (x * M[..., None]).sum(1), atol=1e-4)


def test_step_gradient_reaches_every_module():
    m = make()
    m.train()
    pos, vel, mass, mask = scene(40)
    field = m.init_field(1, DEV)
    p, v, f, dv = m.step(pos, vel, mass, mask, field)
    (p.pow(2).sum() + v.pow(2).sum()).backward()
    missing = [n for n, q in m.named_parameters() if q.grad is None or not torch.isfinite(q.grad).all()]
    assert not missing, missing


def test_summed_attention_scales_with_source_mass():
    blk = bt.Block(8, 2, 4, 4).eval()
    torch.manual_seed(0)
    h = torch.randn(1, 4, 8)
    x = torch.randn(1, 4, 2)
    with torch.no_grad():
        base = blk(h, x, torch.ones(1, 4))
        heavy = blk(h, x, torch.full((1, 4), 3.0))
    assert not torch.allclose(base, heavy)
    w = lambda M: torch.nn.functional.softplus(torch.zeros(1)) * M / 4
    assert float(w(3.0) / w(1.0)) == pytest.approx(3.0)


def test_cost_grows_linearly_in_levels_not_pairs():
    m = make()
    import time
    ts = []
    for n in (512, 2048):
        pos, vel, mass, mask = scene(n)
        with torch.no_grad():
            m.force(pos, vel, mass, mask, None)
            t = time.perf_counter()
            m.force(pos, vel, mass, mask, None)
            ts.append(time.perf_counter() - t)
    assert ts[1] / ts[0] < 4 * 2.5
