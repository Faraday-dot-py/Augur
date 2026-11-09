import torch

from scripts import train_scatter_solar_stability as tr
from tests.test_scatter_solar_moon_arms import aux_data, run


def sched(ramp, iters=6000):
    return [tr.regime_at(i, iters, tr.parse_ramp(ramp), 12, 4, 30, 256) for i in range(iters)]


def changes(ramp, iters=6000):
    out = []
    for i, x in enumerate(sched(ramp, iters)):
        if not out or out[-1][1:] != x:
            out.append((i, *x))
    return out


def test_n_plus_k_equals_existing_staircase():
    a = [(i, k, b) for i, k, b, _ in changes("k,n")]
    assert a == tr.staircase_schedule(6000, 4, 30, 12)


def test_k_alone_equals_existing_linear_ramp():
    for i in range(6000):
        assert tr.regime_at(i, 6000, ("k",), 12, 4, 30, 256) == (tr.k_at(i, 6000, 4, 30), 1.0, 256)


def test_control_is_full_from_start():
    assert set(sched("none")) == {(30, 1.0, 256)}


def test_unramped_factors_stay_full_and_ramped_end_full():
    for r in ("n", "m", "k,m", "n,m", "k,n,m"):
        s = sched(r)
        assert s[-1] == (30, 1.0, 256)
        ramp = tr.parse_ramp(r)
        assert (s[0][0] == 4) == ("k" in ramp) and (s[0][1] == 0.0) == ("n" in ramp) and (s[0][2] == 64) == ("m" in ramp)


def test_one_factor_moves_per_subphase():
    prev = sched("k,n,m")[0]
    moved = 0
    for x in sched("k,n,m")[1:]:
        d = sum(a != b for a, b in zip(x, prev))
        assert d <= 1
        moved += d
        prev = x
    assert moved >= 3 * 11 - 3


def test_monotone():
    for r in ("n", "m", "k,n", "k,m", "n,m", "k,n,m"):
        s = sched(r)
        for c in range(3):
            v = [x[c] for x in s]
            assert v == sorted(v)


def test_mass_rank_narrowest_first():
    M = torch.tensor([[10.0, 1.0, 0.0], [10.0, 1e-4, 1e-3], [10.0, 5.0, 0.0]])
    assert tr.mass_rank(M).tolist() == [2, 0, 1]


def test_regime_runs_and_differs(tmp_path):
    ends = {r: run(tmp_path, ramp=r, curriculum_stages=4, fresh=True) for r in ("none", "k", "n", "m", "k,n", "k,m", "n,m", "k,n,m")}
    assert len({round(v, 6) for v in ends.values()}) >= 6
    assert abs(ends["k,n"] - run(tmp_path, curriculum_stages=4, fresh=True)) < 1e-7


def test_k_alone_matches_default_training(tmp_path):
    assert abs(run(tmp_path, ramp="k", fresh=True) - run(tmp_path, fresh=True)) < 1e-7
