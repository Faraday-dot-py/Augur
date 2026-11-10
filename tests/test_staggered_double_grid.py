import argparse
import json

import pytest
import torch

from scripts import plateau_guard as pg
from scripts import scatter_field as sf
from scripts import staggered_double_grid_verdict as vd
from scripts import train_scatter_field as tsf


def make(**kw):
    torch.manual_seed(4738)
    m = sf.ScatterField(grid=32, extent=32.0, net="unet", levels=3, width=8, kernel=True, pp=2.0, split=True, potential=True, verlet=True, momfix=False, **kw)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def scene(n=40, seed=4738):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(1, n, 2, generator=g) - 0.5) * 20
    vel = torch.randn(1, n, 2, generator=g) * 0.5
    one = torch.ones(1, n)
    return pos, vel, one, one.clone()


def test_quad_mass_conserved_and_fastio_matches():
    m = make(deposit="quad")
    pos, vel, mass, mask = scene()
    s = m.scatter(pos, vel, mass, mask)
    assert torch.allclose(s[:, 0].sum() * m.h ** 2, mass.sum(), rtol=1e-5)
    m.opt.add("fastio")
    s2 = m.scatter(pos, vel, mass, mask)
    assert torch.allclose(s, s2, atol=1e-6)
    f = torch.randn(1, 3, 32, 32)
    assert torch.allclose(m.gather(f, pos), sf.ScatterField.gather(m, f, pos))
    m.opt.discard("fastio")
    assert torch.allclose(m.gather(f, pos), m.gather(f, pos))


def test_quad_gather_reproduces_linear_field_and_cic_unchanged():
    m = make(deposit="quad")
    pos = torch.rand(1, 50, 2) * 10 - 5
    x = (torch.arange(32) + 0.5) * m.h - m.extent / 2
    f = (2.0 * x[None, :] + 3.0 * x[:, None] + 1.0)[None, None]
    g = m.gather(f, pos)[0, :, 0]
    assert torch.allclose(g, 2.0 * pos[0, :, 0] + 3.0 * pos[0, :, 1] + 1.0, atol=1e-4)
    c = make()
    assert c.deposit == "cic" and not c.dual
    assert c.init_field(1, "cpu")[0].shape[1] == c.cf


def test_dual_equals_mean_of_two_shifted_single_passes():
    d, s = make(deposit="quad", dual=True), make(deposit="quad")
    s.load_state_dict(d.state_dict())
    pos, vel, mass, mask = scene()
    fd = d.init_field(1, "cpu")[0]
    assert fd.shape[1] == 2 * d.cf
    f0 = s.init_field(1, "cpu")[0]
    with torch.no_grad():
        ad, fnew = d.force(pos, vel, mass, mask, fd)
        a0, g0 = s.force(pos, vel, mass, mask, f0)
        a1, g1 = s.force(pos + s.h / 2, vel, mass, mask, f0)
    assert torch.allclose(ad, 0.5 * (a0 + a1), atol=1e-5)
    assert fnew.shape == fd.shape


def test_dual_rollout_finite_and_state_shapes():
    d = make(deposit="quad", dual=True)
    pos, vel, mass, mask = scene()
    f = d.init_field(1, "cpu")
    with torch.no_grad():
        for _ in range(3):
            pos, vel, f, _ = d.step(pos, vel, mass, mask, f)
    assert torch.isfinite(pos).all() and f[0].shape[1] == 2 * d.cf


def args_for(tmp_path, iters, **kw):
    return argparse.Namespace(device="cpu", seed=4738, batch=2, k_start=2, k_end=2, steps=6, iters=iters, time_budget=0.0, lr=1e-3, log_every=2, val_every=2, val_scenes=2,
                              long_windows=0, eps=0.5, save_on_sigterm=False, **kw)


def tiny_tensors():
    g = torch.Generator().manual_seed(1)
    P = torch.randn(4, 7, 5, 2, generator=g)
    V = torch.randn(4, 7, 5, 2, generator=g) * 0.1
    return P, V, torch.ones(4, 5), torch.ones(4, 5)


def test_train_resume_and_val_log(tmp_path):
    m = make(deposit="quad", dual=True)
    ck = str(tmp_path / "m.pt")
    tens = tiny_tensors()
    logs = []
    tsf.train(m, tens, args_for(tmp_path, 4), ck, logs.append, val=tens)
    m2 = make(deposit="quad", dual=True)
    tsf.train(m2, tens, args_for(tmp_path, 8), ck, logs.append, val=tens)
    assert any(s.startswith("resumed from it 4") for s in logs)
    rows = [json.loads(x) for x in open(ck + ".val.jsonl")]
    assert [r["it"] for r in rows] == [0, 2, 4, 6]
    assert torch.load(ck, weights_only=False)["it"] == 8


def test_sigterm_checkpoint_then_exit(tmp_path):
    m = make()
    ck = str(tmp_path / "s.pt")
    a = args_for(tmp_path, 10)
    a.save_on_sigterm = True
    import os
    import signal

    seen = []

    def log(s):
        seen.append(s)
        if s.startswith("it 2 "):
            os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(SystemExit) as e:
        tsf.train(m, tiny_tensors(), a, ck, log, val=None)
    assert e.value.code == 143
    assert torch.load(ck, weights_only=False)["it"] == 3


def test_plateau_guard():
    flat = [{"it": i, "val_loss": 1.0} for i in range(0, 100, 2)]
    assert pg.guard(flat, 100)["status"] == "PLATEAUED"
    falling = [{"it": i, "val_loss": 2.0 - i / 100} for i in range(0, 100, 2)]
    assert pg.guard(falling, 100)["status"] == "UNDERTRAINED"
    dup = falling + [{"it": 90, "val_loss": 0.0}]
    assert pg.guard(dup, 100)["status"] == "UNDERTRAINED"
    assert pg.guard(flat[:5], 100)["status"] == "NO-DATA"


def doc(cand, ctl=None):
    ctl = ctl or {"err@5": 1.0, "err@20": 1.0, "err@100": 1.0, "dE@100": 1.0, "probe_half": 1.0, "probe_off": 1.0, "sec_per_tick": 1.0, "finite": 1.0}
    return {"seeds": {s: {"control": ctl, "cand": {**ctl, **cand}} for s in ("9100", "9200", "9300")}}


def test_verdict_lines():
    good = {"err@20": 0.74, "probe_half": 0.5, "dE@100": 1.1, "sec_per_tick": 2.2}
    assert vd.full(doc(good))["verdict"] == "PASS"
    assert vd.full(doc({**good, "err@20": 0.85}))["verdict"] == "INCONCLUSIVE"
    assert vd.full(doc({**good, "err@20": 0.96}))["verdict"] == "KILL"
    assert vd.full(doc({**good, "dE@100": 1.6}))["verdict"] == "KILL"
    assert vd.full(doc({**good, "finite": 0.0}))["verdict"] == "KILL"
    assert vd.full(doc({**good, "sec_per_tick": 2.3}))["verdict"] == "INCONCLUSIVE"
    assert vd.full(doc({**good, "err@20": 0.85}), {"status": "UNDERTRAINED"})["verdict"] == "INCONCLUSIVE-undertrained"
    assert vd.full(doc(good), {"status": "UNDERTRAINED"})["verdict"] == "PASS"


def test_stage1_closure():
    def seed(e2, e4):
        base = {"err@5": 1.0, "err@20": 1.0, "err@100": 1.0, "dE@100": 1.0}
        return {"1": base, "2": {**base, "err@20": e2}, "4": {**base, "err@20": e4}, "probe_half": {"1": 1, "2": 1, "4": 1}, "probe_off": {"1": 1, "2": 1, "4": 1}, "sec_per_tick": {"1": 1, "2": 2, "4": 4}}

    assert vd.stage1({"seeds": {"9100": seed(0.7, 0.6)}})["closes_experiment"] is False
    assert vd.stage1({"seeds": {"9100": seed(0.97, 0.96)}})["closes_experiment"] is True


def test_verdict_smoke_horizons_fall_back():
    r = {"err@5": 1.0, "err@10": 1.0, "dE@10": 1.0, "probe_half": 1.0, "probe_off": 1.0, "sec_per_tick": 1.0, "finite": 1.0}
    d = {"smoke": True, "seeds": {"9100": {"control": dict(r), "cand": dict(r)}}}
    assert vd.full(d)["err@20"] == 1.0
