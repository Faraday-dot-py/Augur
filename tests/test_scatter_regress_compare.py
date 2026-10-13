import copy
import json
from argparse import Namespace

import pytest

from scripts import scatter_regress as sr


def regime():
    return {"mean": {"err": {k: 0.01 for k in sr.TICKS}, "bound_frac_model": 0.5, "bound_frac_true": 0.5, "virial_model": 1.0, "virial_true": 1.0}}


def cons():
    scenes = {f"{q}@{t}": [0.01 * (i + 1) for i in range(6)] for q in ("dE", "dL", "dP") for t in sr.CONS_TICKS}
    for q in ("outgrid", "bound", "vir"):
        scenes[f"{q}@{sr.CONS_STEPS}"] = [0.1] * 6
    model = {"scenes": scenes, "finite": 1.0}
    return {"model": model, "true": model, "gap_scenes": {f"bound@{sr.CONS_STEPS}": [0.01] * 6, f"vir@{sr.CONS_STEPS}": [0.01] * 6}}


def base():
    meta = {"ckpt": "c", "git": "g", "opt": "default", "gpu": "gpu", "torch": "t", "mem_def": sr.MEM_DEF}
    timing = {k: {"tick_ms": 1.0, "peak_mem_mb": 100.0} for k in sr.TIMING_KEYS}
    return {"meta": meta, "regimes": {r: regime() for r in sr.REGIMES}, "timing": timing, "conservation": {r: cons() for r in sr.CONS_REGIMES}}


def run_compare(tmp_path, b, c):
    pb, pc = tmp_path / "b.json", tmp_path / "c.json"
    pb.write_text(json.dumps(b))
    pc.write_text(json.dumps(c))
    with pytest.raises(SystemExit) as e:
        sr.compare(Namespace(baseline=str(pb), candidate=str(pc), floors=str(tmp_path / "none.json")))
    return e.value.code


def test_identical_passes(tmp_path):
    assert run_compare(tmp_path, base(), base()) == 0


def test_two_body_err5_not_gated(tmp_path):
    c = base()
    c["regimes"]["two_body"]["mean"]["err"]["5"] = 1.0
    assert run_compare(tmp_path, base(), c) == 0


@pytest.mark.parametrize("reg,k", [("two_body", "10"), ("expB", "100"), ("bh300", "50")])
def test_err_regression_fails(tmp_path, reg, k):
    c = base()
    c["regimes"][reg]["mean"]["err"][k] = 0.02
    assert run_compare(tmp_path, base(), c) == 1


def test_cons_median_rule(tmp_path):
    c = base()
    c["conservation"]["nr100"]["model"]["scenes"][f"dE@{sr.CONS_STEPS}"] = [0.04] * 6
    assert run_compare(tmp_path, base(), c) == 0
    c["conservation"]["nr100"]["model"]["scenes"][f"dE@{sr.CONS_STEPS}"] = [0.2] * 6
    assert run_compare(tmp_path, base(), c) == 1


def test_bh300_dE_1000_info_only_but_dE_100_gated(tmp_path):
    c = base()
    c["conservation"]["bh300"]["model"]["scenes"][f"dE@{sr.CONS_STEPS}"] = [5.0] * 6
    assert run_compare(tmp_path, base(), c) == 0
    c["conservation"]["bh300"]["model"]["scenes"]["dE@100"] = [5.0] * 6
    assert run_compare(tmp_path, base(), c) == 1


def test_timing_skipped_when_env_differs(tmp_path):
    c = base()
    c["meta"]["opt"] = ""
    for k in c["timing"]:
        c["timing"][k]["tick_ms"] = 10.0
    assert run_compare(tmp_path, base(), c) == 0
    c["meta"]["opt"] = "default"
    assert run_compare(tmp_path, base(), c) == 1


def test_nonfinite_fails(tmp_path):
    c = copy.deepcopy(base())
    c["conservation"]["nr100"]["model"]["finite"] = 0.0
    assert run_compare(tmp_path, base(), c) == 1
