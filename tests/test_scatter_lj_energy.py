import torch

from scripts import scatter_lj as sl


def _model(resid=False):
    torch.manual_seed(0)
    m = sl.build_model(energy=True, resid=resid)
    for mod in [m.ppmlp[-1], m.kmlp[-1], m.net.out] + ([m.rnet.out] if resid else []):
        torch.nn.init.normal_(mod.weight, std=0.05)
    return m


def _state(n, box, seed=1):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(1, n, 2, generator=g, dtype=torch.float64) * torch.tensor(box, dtype=torch.float64)
    vel = torch.randn(1, n, 2, generator=g, dtype=torch.float64)
    return pos, vel, torch.ones(1, n, dtype=torch.float64), torch.tensor([box], dtype=torch.float64)


def test_mesh_force_is_gradient_of_energy_and_velocity_blind():
    m = _model().double()
    pt = sl.PeriodicTiles(m)
    box = [14.0, 13.0]
    pos, vel, mask, bx = _state(60, box)
    lay = sl.layout(bx[0])
    assert lay == (1, 1)
    fld = pt.init_field(1, lay, "cpu", torch.float64)
    with torch.no_grad():
        a, _, a_pp, pk = pt.force(pos, vel, mask, bx, lay, fld, parts=True)
        a2, _ = pt.force(pos, -3 * vel, mask, bx, lay, fld)
    assert torch.allclose(a, a2)
    assert float(a.sum(1).abs().max()) < 1e-8
    P = pk["P"]
    K, V = pk["K"], pk["V"]

    def U(Pt):
        return pt.mesh_energy(m.scatter(Pt, V, K, K)[:, :1])

    eps = 1e-4
    for i in (3, 17):
        for d in (0, 1):
            Pp, Pm = P.clone(), P.clone()
            Pp[0, i, d] += eps
            Pm[0, i, d] -= eps
            fd = -float(U(Pp) - U(Pm)) / (2 * eps)
            Pg = P.clone().requires_grad_(True)
            g = torch.autograd.grad(U(Pg), Pg)[0]
            assert abs(fd + float(g[0, i, d])) < 1e-4 * max(1.0, abs(fd)), (fd, float(g[0, i, d]))


def test_residual_zero_init_and_penalty_taken():
    m = sl.build_model(energy=True, resid=True)
    pt = sl.PeriodicTiles(m)
    pos, vel, mask, bx = _state(40, [12.0, 12.0])
    lay = sl.layout(bx[0])
    a, _ = pt.force(pos, vel, mask, bx, lay, pt.init_field(1, lay, "cpu"))
    pen = pt.take_pen()
    assert pen.shape == (2,) and float(pen.abs().sum()) == 0.0
    assert pt.take_pen().shape == (2,)


def test_energy_force_trains():
    m = _model(resid=True)
    pt = sl.PeriodicTiles(m)
    pos, vel, mask, bx = _state(40, [12.0, 12.0])
    lay = sl.layout(bx[0])
    st = (pt.init_field(1, lay, "cpu"), None)
    p, v = pos, vel
    for _ in range(2):
        p, v, st = pt.step(p, v, mask, bx, lay, st)
    loss = (v ** 2).sum() + pt.take_pen().sum()
    loss.backward()
    assert all(torch.isfinite(q.grad).all() for q in m.net.parameters() if q.grad is not None)
    assert m.net.out.weight.grad is not None and float(m.net.out.weight.grad.abs().sum()) > 0
