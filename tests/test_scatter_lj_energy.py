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
    assert pen.shape == (4,) and float(pen.abs().sum()) == 0.0
    assert pt.take_pen().shape == (4,)


def test_vblind_is_velocity_blind_and_asym_pair_penalties():
    torch.manual_seed(0)
    m = sl.build_model(vblind=True, norec=True)
    for mod in (m.ppmlp[-1], m.kmlp[-1], m.net.out):
        torch.nn.init.normal_(mod.weight, std=0.05)
    pt = sl.PeriodicTiles(m)
    pos, vel, mask, bx = _state(50, [12.0, 12.0])
    lay = sl.layout(bx[0])
    fld = torch.randn_like(pt.init_field(1, lay, "cpu"))
    with torch.no_grad():
        a, _ = pt.force(pos, vel, mask, bx, lay, fld)
        a2, _ = pt.force(pos, -2 * vel, mask, bx, lay, torch.zeros_like(fld))
    assert torch.allclose(a, a2)
    ma = sl.build_model(energy=True, asym=True)
    torch.nn.init.normal_(ma.ppmlp_a[-1].weight, std=0.05)
    pa = sl.PeriodicTiles(ma)
    a, _ = pa.force(pos, vel, mask, bx, lay, pa.init_field(1, lay, "cpu"))
    pen = pa.take_pen()
    assert pen[2] > 0 and pen[3] >= 0
    (pen[2] + pen[3]).backward()
    assert float(ma.ppmlp_a[-1].weight.grad.abs().sum()) > 0


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


def test_oracle_pair_matches_lj_and_nomesh_zeroes_mesh():
    from scripts import lj_sim
    torch.manual_seed(1)
    m = sl.build_model(energy=True, oracle=True, nomesh=True)
    pt = sl.PeriodicTiles(m)
    box_lj = torch.tensor([1000.0, 1000.0], dtype=torch.float64)
    g = torch.arange(6, dtype=torch.float64)
    pos_lj = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) * 1.15 + 0.05 * torch.randn(36, 2, dtype=torch.float64)
    a_ref, _, _ = lj_sim.force(pos_lj, box_lj)
    P = (pos_lj / sl.LS)[None]
    K = torch.ones(1, 36, dtype=torch.float64)
    a = pt.pp_acc_lj(P, K)[0] * sl.LS
    assert float((a - a_ref).abs().max()) < 1e-3 * float(a_ref.abs().max())
    pos, vel, mask, bx = _state(50, [12.0, 12.0])
    lay = sl.layout(bx[0])
    acc, _ = pt.force(pos, vel, mask, bx, lay, pt.init_field(1, lay, "cpu"))
    assert torch.isfinite(acc).all()


def test_respa_pair_sub_one_matches_verlet_and_substeps_run():
    torch.manual_seed(2)
    m = sl.build_model(energy=True)
    for mod in (m.ppmlp[-1], m.net.out):
        torch.nn.init.normal_(mod.weight, std=0.05)
    pt = sl.PeriodicTiles(m)
    pos, vel, mask, bx = _state(40, [12.0, 12.0])
    lay = sl.layout(bx[0])
    st0 = (pt.init_field(1, lay, "cpu"), None)
    with torch.no_grad():
        p1, v1, _ = pt.step(pos, vel, mask, bx, lay, st0)
        m.pair_sub = 1
        p2, v2, _ = pt.step_respa(pos, vel, mask, bx, lay, st0)
        m.pair_sub = 4
        p3, v3, _ = pt.step(pos, vel, mask, bx, lay, st0)
    assert torch.allclose(p1, p2, atol=1e-6) and torch.allclose(v1, v2, atol=1e-6)
    assert torch.isfinite(p3).all() and torch.isfinite(v3).all()
