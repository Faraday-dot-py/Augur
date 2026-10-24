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
    assert "spurious_work_across_seam" not in r
    assert abs(r["seam_jump_ratio_model"] - r["seam_jump_ratio_truth"]) < 1e-6


def test_bond_fracture_potential_consistent():
    rng = np.random.default_rng(4738)
    s = sc.fracture_scene(rng)
    b = sc.build_batch([s], CPU)
    ph = b["phys"]
    assert ph.bonds["brk"] == s["law"]["brk"]
    p = (b["pos"] + 0.05 * torch.randn(b["pos"].shape, dtype=torch.float64)).requires_grad_(True)
    g, = torch.autograd.grad(ph.pe(p, b["vel"], b["mass"], b["mask"]).sum(), p)
    a = ph.acc(p.detach(), b["vel"], b["mass"], b["mask"])
    assert torch.allclose(-g, a * b["mass"][..., None], atol=1e-6, rtol=1e-6)


def test_object_scenes_roll_and_have_split():
    for kind in sc.OBJECT_KINDS:
        rng = np.random.default_rng(4738)
        s = sc.make_scene(kind, rng, 256)
        b = sc.build_batch([s], CPU)
        r = tt.rollout(b["phys"], b["pos"], b["vel"], b["mass"], b["mask"], 3, b["dt"], b["sub"])
        assert torch.isfinite(r["pos"]).all(), kind
    rng = np.random.default_rng(4738)
    s = sc.collision_scene(rng, typ="bonded", b_frac=0.0, v_rel=2.0, om=[0.0, 0.0])
    assert s["tags"]["split"] < len(s["pos"])


def test_objects_group_oracle_and_pile_stays_above_floor():
    out = te.g_objects(ti.Oracle(0.1), CPU, 4738, True)
    assert out["spin"]["err"]["20"] < 1e-9
    k = out["collision_sweep"]["bonded_b0.0_v1.0"]["restitution_ke_ratio"]
    assert abs(k["truth"] - k["model"]) < 1e-9
    rng = np.random.default_rng(4738)
    b = sc.build_batch([sc.pile_scene(rng)], CPU)
    r = tt.rollout(b["phys"], b["pos"], b["vel"], b["mass"], b["mask"], 40, 0.05, 10)
    assert float(r["pos"][0, -1, :, 1].min()) > -17.0


def test_grid_renders_to_step_100(tmp_path):
    import os
    Pt = np.random.default_rng(4738).normal(0, 5, (101, 20, 2))
    te.render_grid("t", (Pt, Pt + 0.1, np.ones(20), np.ones(20)), str(tmp_path / "g.png"))
    assert os.path.getsize(tmp_path / "g.png") > 1000


def test_arms_compare_verdict_logic(tmp_path):
    import json
    import types

    from scripts import tiered_arms as ta

    def arm(far, near, ms2000, ms1, ms0125):
        return dict(model="m", per_seed={str(s): dict(far=dict(rel_rms_far_dominated=f, rel_rms_all=f), near=dict(rel_rms=near + 0.001 * i),
                                                      rollout=dict(err={"100": 1.0}),
                                                      cost=dict(n_sweep={"2000": dict(ms=ms2000)}, empty_sweep={"1.0": dict(ms=ms1), "0.125": dict(ms=ms0125)}))
                                   for i, (s, f) in enumerate(zip(ta.HELD, far))})
    json.dump(arm([0.2, 0.21, 0.19], 0.05, 10, 10, 10), open(tmp_path / "arms_B.json", "w"))
    json.dump(arm([0.05, 0.06, 0.05], 0.05, 11, 11, 11), open(tmp_path / "arms_C.json", "w"))
    ta.compare(types.SimpleNamespace(dir=str(tmp_path)))
    assert json.load(open(tmp_path / "arms_compare.json"))["verdict"] == "PASS"
    json.dump(arm([0.2, 0.21, 0.19], 0.05, 11, 11, 11), open(tmp_path / "arms_C.json", "w"))
    ta.compare(types.SimpleNamespace(dir=str(tmp_path)))
    assert json.load(open(tmp_path / "arms_compare.json"))["verdict"] == "KILL"


def test_t0ext_adapter_when_branch_code_present():
    import pytest
    pytest.importorskip("scripts.scatter_tier0")
    from scripts import scatter_tier0 as t0m
    from scripts import tiered_adapters as ta
    torch.manual_seed(4738)
    m = sf.ScatterField(grid=32, extent=16.0, net="local", recurrent=True, hidden_ch=0, dt=0.1, width=8, layers=2, momfix=True)
    t0m.attach_tier0(m, "ext=6,extapply,bnd,halo=2")
    m = m.double()
    ad = ta.Tier0ExtAdapter(m, "t0", dtype=torch.float64)
    assert {"ext", "occ"} <= set(ad.caps)
    pos, vel, mass, mask = _rand(20, spread=4.0)
    gen = torch.Generator().manual_seed(4738)
    ctx = dict(ext=tt.ExtField.random(1, gen, CPU), t=0.0, bound=[dict(kind="wall", lo=np.array([-6.0, -6.0]), hi=np.array([6.0, 6.0]))])
    p2, v2, st = ad.step(pos, vel, mass, mask, ad.init_state(1, CPU), ctx)
    assert torch.isfinite(p2).all() and torch.isfinite(v2).all()


def test_holes_respa_verdict_logic():
    from scripts import tiered_holes as th
    ok = {1: dict(err=[1.0, 1.1, 0.9], dE=[0.01, 0.01, 0.01], ms=[10, 10, 10]), 2: dict(err=[1.02, 1.1, 0.95], dE=[0.01, 0.01, 0.01], ms=[7, 7, 7])}
    assert th.respa_verdict(ok)["verdict"] == "PASS"
    bad = {1: ok[1], 2: dict(err=[2.0, 2.3, 2.1], dE=[0.01] * 3, ms=[7] * 3)}
    assert th.respa_verdict(bad)["verdict"] == "KILL"
    slow = {1: ok[1], 2: dict(err=[1.0, 1.1, 0.9], dE=[0.01] * 3, ms=[10] * 3)}
    assert th.respa_verdict(slow)["verdict"] == "MIXED"


def test_holes_respa_runs_with_exact_multirate_stub():
    from scripts import tiered_holes as th

    class Stub(ti.Oracle):
        def step_multirate(self, pos, vel, mass, mask, state, m):
            for _ in range(m):
                pos, vel, state = ti.Oracle.step(self, pos, vel, mass, mask, state, dict(phys=tt.Physics(grav=dict(eps=tt.EPS))))
            return pos, vel, state
    out = th.run_respa(Stub(0.1), CPU, [9100, 9200], T=8, ms=(1, 2))
    assert max(out["rows"][2]["err"]) < 1e-6


def test_holes_exposure_verdict_logic():
    from scripts import tiered_holes as th

    def arm(e, d, z):
        return {str(s): dict(err100=e + 0.01 * i, dE100=d + 0.001 * i, tier0_bias_absz=z, gap={"50": dict(ratio=1.0 + 0.02 * i)}) for i, s in enumerate(th.HELD)}
    res = dict(S=arm(5.0, 0.05, 1.0), R=arm(3.0, 0.02, 1.0), J=arm(3.0, 0.02, 4.0), N=arm(5.0, 0.05, 1.0))
    v = th.exposure_verdict(res)
    assert v["R"]["verdict"] == "PASS" and v["J"]["verdict"] == "KILL" and v["N"]["verdict"] == "NO_GAIN"
    assert v["S_gap_ratio_step50"]["teacher_forcing_suffices"] in (True, False)


def test_holes_sparse_verdict_logic():
    from scripts import tiered_holes as th
    r = dict(equiv_rel_rms=1e-5, ms_sparse_0125=4.0, ms_dense_1=10.0, refine=dict(err50_base=[1.0, 1.1, 1.2], err50_refined=[0.7, 0.8, 0.85], far_base=[0.4] * 3, far_refined=[0.4] * 3))
    v = th.sparse_verdict(r)
    assert v["sparse"] == "PASS" and v["refine"]["verdict"] == "PASS"
    assert th.sparse_verdict(dict(r, equiv_rel_rms=0.5))["sparse"] == "KILL"


def test_holes_recurrent_verdict_and_refresh_runs():
    from scripts import tiered_holes as th
    r = dict(reset=dict(err50=[2.0, 2.1, 2.2]), persist=dict(err50=[1.5, 1.6, 1.4], ms=10.0, seam_jump=2.0),
             refresh=dict(err50=[1.0, 1.1, 0.9], ms=11.0, seam_jump=0.5), shift=dict(half=0.003, floor=0.002))
    v = th.recurrent_verdict(r)
    assert v["refresh"]["verdict"] == "PASS" and v["shift_invariance"]["verdict"] == "PASS"
    ad = _tiny_scatter()
    tl = ti.Tiled(ad, 64.0, sparse=False, persist=True, refresh=True)
    pos, vel, mass, mask = _rand(40, spread=20.0)
    st = tl.init_state(1, CPU)
    for _ in range(2):
        a, st = tl.force(pos, vel, mass, mask, st, None)
    assert torch.isfinite(a).all()


class _Calib(ti.Oracle):
    """Oracle plus a deterministic force error proportional to the true acceleration; error() reports it (noisy) or noise only."""

    def __init__(self, informative=True):
        super().__init__(0.1)
        self.informative = informative

    def force(self, pos, vel, mass, mask, state, ctx=None):
        a, st = super().force(pos, vel, mass, mask, state, ctx)
        return a * (1 + 0.3 * torch.sin(7.0 * pos[..., 0:1] + 3.0 * pos[..., 1:2])), st

    def error(self, pos, vel, mass, mask, state, ctx=None):
        true = tt.grav_acc(pos, mass, mask)
        err = (self.force(pos, vel, mass, mask, state, ctx)[0] - true).norm(dim=-1)
        g = torch.Generator().manual_seed(4738)
        noise = torch.rand(err.shape, generator=g, dtype=err.dtype)
        return err * (1 + 0.1 * noise) if self.informative else noise


def test_holes_calibration_pass_and_kill():
    from scripts import tiered_holes as th
    good = th.run_calibration(_Calib(True), CPU, [4738, 9100])
    assert good["verdict"] == "PASS", good
    bad = th.run_calibration(_Calib(False), CPU, [4738, 9100])
    assert bad["verdict"] == "KILL", bad
    assert th.run_calibration(ti.ConstVel(), CPU, [4738])["verdict"].startswith("skipped")
