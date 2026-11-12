import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import bit_output_64_grid_long_range as bo  # noqa: E402
import bit_output_encoding_fixes as ef  # noqa: E402


class Args:
    eval_scenes, rollout_scenes = 512, 256


def test_toy_path_reproduces_oracle_and_features():
    torch.manual_seed(0)
    geo = bo.Geo(8, 48, "short", (), 8, 0.05, 1.0)
    e = bo.make_eval(geo, 4739, Args, "cpu")
    ref = ef.references(ef.make_eval(4738, "cpu"))["particle_force_rel_err"]["oracle_cell_mean"]
    got = bo.references(geo, e)["oracle_cell_mean_F"]
    assert got == pytest.approx(ref, abs=1e-6)
    assert got == pytest.approx(0.596, abs=0.01)
    feats, n = geo.features(e["pos"][:4])
    from bit_force_smoke import features as toy_features, centered_windows
    grid, n0 = toy_features(e["pos"][:4])
    assert torch.allclose(feats, centered_windows(grid), atol=1e-6)
    assert torch.equal(n, n0)


def test_long_range_geometry():
    geo = bo.Geo(64, 40, "long", (2, 4, 8), 8, 0.05, 1.0)
    assert geo.din == 3 * 64 * 4
    pos = geo.sample(2, torch.Generator().manual_seed(1), "cpu")
    feats, n = geo.features(pos)
    assert feats.shape == (2, 64 * 64, geo.din) and n.sum().item() == 80
    f = geo.force(pos)
    assert f.shape == pos.shape and torch.isfinite(f).all()
    assert torch.allclose(f.sum(1), torch.zeros(2, 2), atol=1e-5)


def test_coarse_copy_is_constant_within_parent():
    geo = bo.Geo(64, 30, "long", (8,), 8, 0.05, 1.0)
    pos = geo.sample(1, torch.Generator().manual_seed(2), "cpu")
    feats, _ = geo.features(pos)
    coarse = feats[0, :, 192:].reshape(8, 8, 8, 8, -1)
    assert torch.equal(coarse[0, 0], coarse[0, 7]) and torch.equal(coarse[3, 2], coarse[3, 5])


def test_soft_decode_matches_hard_on_saturated_logits():
    codec = ef.Codec(8, 0.01)
    dp = torch.tensor([[0.3, -0.5], [0.0, 0.77]])
    bits = codec.encode(dp)
    assert torch.allclose(codec.soft_decode((bits - 0.5) * 80), codec.decode(bits), atol=1e-5)


def test_plateau_boundary_and_missing_window():
    steps = 100
    flat = [(s, 1.0) for s in range(10, 101, 10)]
    assert not bo.plateau(flat, steps)["undertrained"]
    falling = [(s, 1.0 if s <= 90 else 0.96) for s in range(10, 101, 10)]
    assert bo.plateau(falling, steps)["undertrained"]
    nearly = [(s, 1.0 if s <= 90 else 0.971) for s in range(10, 101, 10)]
    assert not bo.plateau(nearly, steps)["undertrained"]
    with pytest.raises(SystemExit):
        bo.plateau([(50, 1.0)], steps)


def _res(fb, fs, rb, rs, up=False):
    r = {"runs": {}, "val": {}}
    for s in (1, 2):
        mk = lambda F, R: {"F": F, "R": {"err@20": R}}
        r["runs"][f"4738/{s}"] = {"float": mk(1.0, 1.0), "bits_soft": mk(fs, rs), "bits_hard": mk(fb, rb)}
    flat = [(s, 1.0) for s in range(10, 101, 10)]
    fall = [(s, 1.0 if s <= 90 else 0.9) for s in range(10, 101, 10)]
    r["val"]["4738/float"] = flat
    r["val"]["4738/bits"] = fall if up else flat
    return r


def test_scorer_labels_mechanical():
    assert bo.score(_res(2, 1.5, 2, 1.5), 4738, [1, 2], 100)["label"] == "PASS"
    assert bo.score(_res(2, 1.51, 2, 1.0), 4738, [1, 2], 100)["label"] == "INCONCLUSIVE"
    assert bo.score(_res(2, 1.0, 2, 3.01), 4738, [1, 2], 100)["label"] == "KILL"
    assert bo.score(_res(2, 1.0, 2, 3.01, up=True), 4738, [1, 2], 100)["label"] == "INCONCLUSIVE-undertrained"
    assert bo.score(_res(2, 1.0, 2, 1.0, up=True), 4738, [1, 2], 100)["label"] == "PASS"
