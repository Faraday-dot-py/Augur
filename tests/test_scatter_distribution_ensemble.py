import json

import numpy as np
import torch

from scripts import scatter_distribution_ensemble as de
from scripts import scatter_field as sf


def test_exact_batch_matches_reference_integrator():
    rng = np.random.default_rng(4738)
    pos = rng.normal(0, 5, (20, 2))
    vel = rng.normal(0, 0.3, (20, 2))
    P, V = sf.rollout_mass_torch(pos, vel, np.ones(20), 10, torch.device("cpu"))
    out = de.exact_batch(torch.tensor(pos)[None].repeat(2, 1, 1), torch.tensor(vel)[None].repeat(2, 1, 1), 10, torch.device("cpu"), keep=(5,))
    assert np.abs(out[10][0][0].numpy() - P[10]).max() < 1e-9
    assert np.abs(out[10][1][1].numpy() - V[10]).max() < 1e-9
    assert set(out) == {0, 5, 10}


def test_perturb_scale_and_profiles_conserve_bodies():
    rng = np.random.default_rng(1)
    pos = rng.normal(0, 8, (200, 2))
    pk = de.perturb(pos, 1e-3, 64, rng)
    L = np.sqrt(((pos - pos.mean(0)) ** 2).sum(-1).mean())
    assert abs(np.std(pk - pos[None]) / (1e-3 * L) - 1) < 0.05
    cnt, rms = de.profiles(pos, rng.normal(0, 1, (200, 2)))
    assert cnt.sum() == 200 and np.isnan(rms[cnt == 0]).all() and (rms[cnt > 0] > 0).all()
    far = np.array([[1000.0, 0.0], [0.0, 0.0], [0.1, 0.0]])
    assert de.profiles(far, np.zeros((3, 2)))[0][-1] >= 1


def test_inside_band_and_pooling():
    rng = np.random.default_rng(2)
    ens = rng.normal(10.0, 1.0, (200, 4))
    ens[:, 3] = np.nan
    assert de.inside_band(np.array([10.0, 10.5, 12.5, 10.0]), ens).tolist() == [True, True, False, False]
    cnt = rng.poisson(20, (16, de.NB)).astype(float)
    rms = rng.normal(1.0, 0.1, (16, de.NB))
    s = de.score_scene((cnt.mean(0), rms.mean(0)), (cnt, rms))
    assert s["density"]["inside"] == s["density"]["scored"] == de.NB
    shifted = de.score_scene((cnt.mean(0) + 100.0, rms.mean(0)), (cnt, rms))
    assert shifted["density"]["inside"] == 0 and shifted["speed"]["inside"] == shifted["speed"]["scored"]
    p = de.pool([s, shifted])
    assert abs(p["density"]["frac"] - 0.5) < 1e-12 and 0.8 < p["density"]["self_frac"] <= 1.0


def test_control_scoring_end_to_end_resumes(tmp_path, monkeypatch):
    from scripts import score_scatter_distribution_control as sc
    from scripts import train_scatter_field as tsf
    from scripts import train_scatter_timestep as tt

    monkeypatch.setitem(tsf.VARIANTS, "tiny_ctrl", dict(net="unet", kernel=True, pp=2.0, split=True, potential=True, verlet=True, grid=32, width=8))
    monkeypatch.setattr(sc.cf, "load_model", lambda exp, ckpt, dt, dev: tt.build_model("tiny_ctrl", dev).eval())
    argv = ["--ckpt", "unused", "--out-dir", str(tmp_path), "--device", "cpu", "--steps", "21", "--scenes", "2", "--members", "3", "--rels", "1e-4"]
    assert sc.main(argv + ["--seeds", "9100,9200,9300,12000,12100,12200"]) == 0
    s = json.load(open(tmp_path / "summary.json"))
    assert set(s["summary"]) == {"pre_registered", "fresh"}
    r = json.load(open(tmp_path / "seed9100_rel0.0001.json"))
    assert r["members"] == 3 and 0.0 <= r["density"]["frac"] <= 1.0 and r["err5"] > 0
    mtime = (tmp_path / "seed9100_rel0.0001.json").stat().st_mtime
    assert sc.main(argv + ["--seeds", "9100,9200,9300,12000,12100,12200"]) == 0
    assert (tmp_path / "seed9100_rel0.0001.json").stat().st_mtime == mtime
