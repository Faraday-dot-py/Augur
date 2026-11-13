import math

import pytest
import torch

from scripts import scatter_field as sf

KW = dict(grid=32, extent=64.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, momfix=True, levels=3)


def stock(seed=4738):
    torch.manual_seed(seed)
    m = sf.ScatterField(**KW)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def cand_from(e):
    c = sf.ScatterField(pairctx=True, **KW)
    sf.load_compat(c, e.state_dict())
    return c.eval()


def scene(n=40, seed=4738):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(1, n, 2, generator=g) - 0.5) * 6
    vel = torch.randn(1, n, 2, generator=g) * 0.5
    one = torch.ones(1, n)
    return pos, vel, one, one.clone()


def pair(m, pos, vel, mass, mask):
    return m.pp_acc(pos, mass, mask, 16, vel=vel)


def test_flag_off_keeps_keys_and_shapes():
    e = stock()
    assert not e.pairctx
    assert e.ppmlp[0].weight.shape == (64, 2) and e.ppmlp[-1].weight.shape == (1, 64)
    plain = sf.ScatterField(**KW)
    assert list(plain.state_dict()) == list(e.state_dict())


def test_flag_off_force_matches_vel_independent_stock():
    e = stock()
    pos, vel, mass, mask = scene()
    with torch.no_grad():
        a = e.pp_acc(pos, mass, mask, 16)
        b = e.pp_acc(pos, mass, mask, 16, vel=vel * 3)
    assert torch.equal(a, b)


def test_init_equals_stock_and_tangential_zero():
    e = stock()
    c = cand_from(e)
    pos, vel, mass, mask = scene()
    assert float(c.ppmlp[-1].weight[1].abs().sum()) == 0.0 and float(c.ppmlp[-1].bias[1]) == 0.0
    assert float(c.ppmlp[0].weight[:, 2:].abs().sum()) == 0.0
    with torch.no_grad():
        assert torch.allclose(pair(c, pos, vel, mass, mask), e.pp_acc(pos, mass, mask, 16), atol=1e-6, rtol=1e-5)
        a_e, _ = e.force(pos, vel, mass, mask, e.init_field(1, "cpu")[0])
        a_c, _ = c.force(pos, vel, mass, mask, c.init_field(1, "cpu")[0])
    assert torch.allclose(a_e, a_c, atol=1e-5, rtol=1e-4)


def test_pair_force_rotation_equivariant():
    c = cand_from(stock())
    for p in c.ppmlp.parameters():
        torch.nn.init.normal_(p, std=0.2)
    pos, vel, mass, mask = scene()
    th = 0.7
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    with torch.no_grad():
        a = pair(c, pos, vel, mass, mask)
        b = pair(c, pos @ R.T, vel @ R.T, mass, mask)
    assert torch.allclose(a @ R.T, b, atol=1e-4, rtol=1e-4)


def test_tangential_branch_depends_on_velocity_and_density():
    c = cand_from(stock())
    c.ppmlp[-1].weight.data[1] = torch.randn(64) * 0.1
    c.ppmlp[0].weight.data[:, 2:] = torch.randn(64, 3) * 0.3
    pos, vel, mass, mask = scene()
    with torch.no_grad():
        assert not torch.allclose(pair(c, pos, vel, mass, mask), pair(c, pos, vel * 0, mass, mask))


def test_checkpoint_round_trip():
    c = cand_from(stock())
    for p in c.ppmlp.parameters():
        torch.nn.init.normal_(p, std=0.2)
    d = sf.ScatterField(pairctx=True, **KW).eval()
    d.load_state_dict(c.state_dict())
    pos, vel, mass, mask = scene()
    with torch.no_grad():
        assert torch.equal(pair(c, pos, vel, mass, mask), pair(d, pos, vel, mass, mask))


def test_non_dense_mode_fails_loudly():
    c = cand_from(stock())
    c.opt = {"cell"}
    pos, vel, mass, mask = scene(n=600)
    pos = pos * 10
    with pytest.raises(NotImplementedError):
        pair(c, pos, vel, mass, mask)
    c.opt = set()
    with pytest.raises(NotImplementedError):
        c.pp_acc(pos[:, :40], mass[:, :40], mask[:, :40], 16)


def test_stats_record_ratio():
    c = cand_from(stock())
    c.ppmlp[-1].weight.data[1] = torch.randn(64) * 0.1
    c.pp_stats = {}
    pos, vel, mass, mask = scene()
    with torch.no_grad():
        pair(c, pos, vel, mass, mask)
    assert c.pp_stats["tan_sq"] > 0 and c.pp_stats["rad_sq"] > 0


def test_tangential_penalty_off_by_default_and_matches_metric():
    c = cand_from(stock())
    assert c.pen_acc is None
    c.ppmlp[-1].weight.data[1] = torch.randn(64) * 0.1
    pos, vel, mass, mask = scene()
    c.pp_stats = {}
    c.pen_acc = []
    pair(c, pos, vel, mass, mask)
    assert len(c.pen_acc) == 1
    assert math.isclose(float(c.pen_acc[0]), c.pp_stats["tan_sq"] / c.pp_stats["rad_sq"], rel_tol=1e-4)


def test_tangential_penalty_gradient_reaches_tangential_row_only():
    c = cand_from(stock())
    c.ppmlp[-1].weight.data[1] = torch.randn(64) * 0.1
    pos, vel, mass, mask = scene()
    c.pen_acc = []
    pair(c, pos, vel, mass, mask)
    c.pen_acc[0].backward()
    g = c.ppmlp[-1].weight.grad
    assert float(g[1].abs().sum()) > 0 and float(g[0].abs().sum()) == 0.0


def test_score_lines_replication_and_penalty():
    from scripts import pair_context_eval as pe

    def arm(e20, tan=None, mom=1.0):
        v = [(i, 1.0) for i in range(0, 100, 5)]
        a = {"err20_mean": e20, "momentum_error_mean": mom, "validation": v}
        if tan is not None:
            a["tangential_over_radial"] = tan
        return a
    res = {"arms": {"ctrl-s1": arm(1.0), "cand-s1": arm(0.85, 0.08), "ctrl-s2": arm(1.0), "cand-s2": arm(0.93, 0.08)}}
    assert pe.score_lines(res, [1, 2], 100, "seed-replication")["label"] == "INCONCLUSIVE"
    res["arms"]["cand-s2"]["err20_mean"] = 0.951
    assert pe.score_lines(res, [1, 2], 100, "seed-replication")["label"] == "KILL"
    res["arms"]["cand-s2"]["err20_mean"] = 0.90
    assert pe.score_lines(res, [1, 2], 100, "seed-replication")["label"] == "PASS"
    assert pe.score_lines(res, [1, 2], 100, "tangential-penalty")["label"] == "INCONCLUSIVE"
    one = {"arms": {"ctrl-s1": arm(1.0), "cand-s1": arm(0.85, 0.06)}}
    assert pe.score_lines(one, [1], 100, "tangential-penalty")["label"] == "INCONCLUSIVE"
    one["arms"]["cand-s1"]["tangential_over_radial"] = 0.049
    assert pe.score_lines(one, [1], 100, "tangential-penalty")["label"] == "PASS"
