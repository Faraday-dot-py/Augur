import torch

from scripts import scatter_lj as sl


def _model():
    torch.manual_seed(0)
    m = sl.build_model()
    for mod in (m.ppmlp[-1], m.kmlp[-1], m.net.out):
        torch.nn.init.normal_(mod.weight, std=0.05)
    return m


def _state(n, box, seed=1):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(1, n, 2, generator=g, dtype=torch.float64) * torch.tensor(box, dtype=torch.float64)
    vel = torch.randn(1, n, 2, generator=g, dtype=torch.float64)
    return pos, vel, torch.ones(1, n, dtype=torch.float64), torch.tensor([box], dtype=torch.float64)


def _brute_pp(m, pos, box):
    rel = pos[0][None, :, :] - pos[0][:, None, :]
    rel = rel - box[0] * torch.round(rel / box[0])
    r = torch.sqrt((rel ** 2).sum(-1) + 1e-8)
    bad = torch.eye(len(r), dtype=torch.bool)
    rf = r.float()
    g = m.ppmlp(torch.stack([rf, torch.log(rf + 0.05)], -1))[..., 0].double()
    w = g * ((1 - (r / m.pp).clamp(max=1.0) ** 2) ** 2) * (~bad)
    return (w[..., None] * rel / r[..., None]).sum(1)


def test_every_body_owned_once_and_pp_matches_min_image():
    m = _model()
    pt = sl.PeriodicTiles(m, knn=40)
    box = [45.0, 38.0]
    pos, vel, mask, bx = _state(300, box)
    lay = sl.layout(bx[0])
    assert lay == (3, 2)
    with torch.no_grad():
        a, f, a_pp, pk = pt.force(pos, vel, mask, bx, lay, pt.init_field(1, lay, "cpu"), parts=True)
    assert len(pk["s"]) == 300 and len(set(pk["b"].tolist())) == 300
    ref = _brute_pp(m, pos, bx)
    ref = ref - ref.mean(0)
    pp = a_pp[0] - a_pp[0].mean(0)
    assert (pp - ref).abs().max() < 2e-4 * max(1.0, ref.abs().max())


def test_layout_independent_pp_and_zero_momentum():
    m = _model()
    pt = sl.PeriodicTiles(m, knn=40)
    pos, vel, mask, bx = _state(150, [18.0, 18.0], seed=3)
    with torch.no_grad():
        a1, _, p1, _ = pt.force(pos, vel, mask, bx, (1, 1), pt.init_field(1, (1, 1), "cpu"), parts=True)
        a2, _, p2, _ = pt.force(pos, vel, mask, bx, (2, 2), pt.init_field(1, (2, 2), "cpu"), parts=True)
    assert (p1 - p2).abs().max() < 1e-4
    assert a1.sum(1).abs().max() < 1e-4 and a2.sum(1).abs().max() < 1e-4


def test_step_wraps_and_is_finite():
    m = _model()
    pt = sl.PeriodicTiles(m)
    pos, vel, mask, bx = _state(80, [25.0, 22.0], seed=5)
    lay = sl.layout(bx[0])
    st = (pt.init_field(1, lay, "cpu"), None)
    with torch.no_grad():
        for _ in range(3):
            pos, vel, st = pt.step(pos, vel, mask, bx, lay, st)
    assert torch.isfinite(pos).all() and torch.isfinite(vel).all()
    assert (pos >= 0).all() and (pos < bx[:, None, :]).all()
