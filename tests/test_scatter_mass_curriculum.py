import os
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts import scatter_field as sf
from scripts import scatter_mass_curriculum_data as cd
from scripts import train_scatter_mass_curriculum as tm

KW = dict(grid=32, extent=16.0, net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, width=8)


def tiny(**kw):
    torch.manual_seed(0)
    m = sf.ScatterField(**{**KW, **kw})
    return m


def randomise(m):
    g = torch.Generator().manual_seed(1)
    for p in m.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.05
    return m


def scene(n=12, seed=0, ratio=1.0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(1, n, 2, generator=g, dtype=torch.float64) * 8 - 4
    vel = torch.randn(1, n, 2, generator=g, dtype=torch.float64) * 0.3
    mass = torch.ones(1, n, dtype=torch.float64)
    mass[0, 0] = ratio
    return pos.float(), vel.float(), mass.float(), torch.ones(1, n)


def accel(m, pos, vel, mass, mask):
    f = m.init_field(1, "cpu")[0]
    return m.force(pos, vel, mass, mask, f)[0]


def test_massfeat_warm_start_is_identity_for_unit_mass():
    base = randomise(tiny(norm="mass"))
    mf = tiny(norm="mass", massfeat=True)
    sf.load_compat(mf, base.state_dict())
    p, v, m, k = scene()
    assert torch.allclose(accel(base, p, v, m, k), accel(mf, p, v, m, k), atol=1e-5)


def test_acceleration_scales_with_mass_unit():
    m = randomise(tiny(norm="gmean", massfeat=True))
    p, v, mass, k = scene(ratio=50.0)
    a1 = accel(m, p, v, mass, k)
    a2 = accel(m, p, v, 3.0 * mass, k)
    assert torch.allclose(a2, 3.0 * a1, rtol=1e-3, atol=1e-5)


def test_mass_unit_override_is_scene_constant():
    m = randomise(tiny(norm="gmean", massfeat=True))
    p, v, mass, k = scene(ratio=50.0)
    ns = float(torch.exp(torch.log(mass).mean()))
    a_auto = accel(m, p, v, mass, k)
    m.mass_unit = ns
    assert torch.allclose(a_auto, accel(m, p, v, mass, k), atol=1e-5)
    m.mass_unit = 2 * ns
    assert not torch.allclose(a_auto, accel(m, p, v, mass, k), atol=1e-5)


def test_heavy_body_cell_stays_finite_and_softclip_is_identity_below_c():
    x = torch.tensor([-3.0, 0.0, 10.0, 64.0])
    assert torch.equal(sf.softclip(x), x)
    assert sf.softclip(torch.tensor(4000.0)) < 400
    m = randomise(tiny(norm="gmean", massfeat=True))
    p, v, mass, k = scene(ratio=1000.0)
    assert torch.isfinite(accel(m, p, v, mass, k)).all()


def test_momentum_conserved_with_mixed_masses():
    m = randomise(tiny(norm="gmean", massfeat=True, momfix=True))
    p, v, mass, k = scene(ratio=1000.0)
    a = accel(m, p, v, mass, k)
    assert (mass[..., None] * a).sum(1).abs().max() < 1e-3 * (mass[..., None] * a.abs()).sum(1).max()


def test_orbit_elements_roundtrip():
    rng = np.random.default_rng(0)
    M, m = 500.0, 5.0
    rel, vel, a, e = cd.planet_orbit(rng, M, m, a=12.0, e=0.2)
    pos = np.stack([np.zeros(2), rel])
    v = np.stack([np.zeros(2), vel])
    am, em = cd.orbit_elements(pos, v, np.array([M, m]))
    assert abs(am[0] - 12.0) / 12.0 < 0.01 and abs(em[0] - 0.2) < 0.02


def test_scene_generators_respect_floor_ratio_and_momentum():
    rng = np.random.default_rng(1)
    for q in cd.STAGES:
        pos, vel, mass = cd.star_planet(rng, q)
        assert mass[0] / mass[1] == pytest.approx(q)
        assert np.abs((mass[:, None] * vel).sum(0)).max() < 1e-8
        r = np.linalg.norm(pos[1] - pos[0])
        _, _, a, _ = cd.planet_orbit(rng, mass[0], mass[1])
        assert a >= cd.a_min(mass[0] + mass[1]) * 0.999
        pos, vel, mass = cd.star_disk(rng, q)
        assert mass[0] == mass.max() and mass[2:].sum() <= 0.1 * mass[0] + 1e-9
        assert np.abs(pos).max() < 32
        assert np.abs((mass[:, None] * vel).sum(0)).max() < 1e-8


def test_scenes_truth_and_metrics_zero_for_truth_model():
    data = cd.scenes(2, 30, 4738, "cpu", 100.0, "planet")
    P, V, M, mask = sf.to_tensors(data, "cpu")
    assert M[0, 0] / M[0, 1] == pytest.approx(100.0)
    # model with all-zero output (ballistic) must show a nonzero element error for an orbiting planet
    m = tiny(norm="gmean", massfeat=True)
    r = cd.orbit_metrics(m, data, "cpu", (20, 30))
    assert r[30]["da_over_a"] > 0 and r[30]["finite"]


def test_verdict_lines():
    def blk(da, dE, de=0.0):
        return {t: dict(da_over_a=da, de=de, pos_err=0.0, dE_model=dE, dE_truth=0.0, dE_model_max=dE, finite=True) for t in cd.__dict__.get("EVAL", tm.EVAL_TICKS)}
    def res(da, dE):
        return dict(planet={q: blk(da, dE) for q in tm.EVAL_Q}, disk={q: blk(da, dE) for q in tm.EVAL_Q}, base={5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0})
    init = res(0, 0)
    assert tm.verdict(res(0.01, 0.01), init)[0] == "PASS"
    assert tm.verdict(res(0.04, 0.01), init)[0] == "INCONCLUSIVE"
    assert tm.verdict(res(0.5, 0.01), init)[0] == "KILL"
    r = res(0.01, 0.01)
    r["base"][10] = 2.0
    assert tm.verdict(r, init)[0] == "KILL"




def tiny_variant():
    return tm.build_model("massfeat_curriculum", "cpu", 0.1)


def test_train_smoke_resume_and_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "EVAL_Q", (10,))
    monkeypatch.setattr(cd, "STAGES", (1, 10))
    monkeypatch.setattr(tm, "evaluate", lambda model, device, **kw: dict(planet={}, disk={}, base={5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0}))
    init = tmp_path / "init.pt"
    torch.save({"model": tiny_variant().state_dict()}, init)
    argv = ["--arm", "massfeat_curriculum", "--init-ckpt", str(init), "--device", "cpu", "--out-dir", str(tmp_path / "r"), "--ckpt-dir", str(tmp_path / "c"),
            "--stage-iters", "2", "--scenes", "4", "--steps", "10", "--batch", "2", "--k-start", "2", "--k-end", "3", "--ckpt-every", "1", "--log-every", "1", "--warmup", "1"]
    tm.stop["flag"] = True
    assert tm.main(argv) == 143
    st = torch.load(tmp_path / "c" / "massfeat_curriculum.pt", weights_only=False)
    assert st["it"] == 0
    tm.stop["flag"] = False
    monkeypatch.setattr(tm, "verdict", lambda r, i: ("PASS", []))
    assert tm.main(argv) == 0
    st = torch.load(tmp_path / "c" / "massfeat_curriculum.pt", weights_only=False)
    assert st["it"] == 4
