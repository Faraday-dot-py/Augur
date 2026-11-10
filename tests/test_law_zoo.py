import numpy as np
import torch

from scripts import law_zoo as lz
from scripts import scatter_far_general as sfar
from scripts import scatter_multi as sm


def test_effective_exponent_of_known_laws():
    assert abs(lz.effective_exponent(sfar.LAWS["pow2"]) - 2.0) < 0.05
    assert abs(lz.effective_exponent(sfar.LAWS["pow1.5"]) - 1.5) < 0.05
    assert abs(lz.effective_exponent(sfar.LAWS["pow3"]) - 3.0) < 0.05
    assert lz.effective_exponent(sfar.LAWS["yuk32"]) > 2.1


def test_zoo_excludes_held_out_neighbourhoods_and_heldout_sits_inside():
    assert lz.held_out_distance(sfar.LAWS["pow1.5"]) < 1e-9 and lz.held_out_distance(sfar.LAWS["yuk32"]) < 1e-9
    assert lz.held_out_distance(sfar.LAWS["pow2"]) > lz.SHAPE_TOL and lz.held_out_distance(sfar.LAWS["pow1"]) > lz.SHAPE_TOL
    zoo, excluded = lz.sample_zoo(60, 4738)
    assert len(zoo) == 60 and excluded >= 0
    assert min(lz.held_out_distance(x) for x in zoo) >= lz.SHAPE_TOL
    assert lz.sample_zoo(60, 4738)[0] == zoo and lz.sample_zoo(60, 4739)[0] != zoo
    held = lz.sample_heldout(20, 4738)
    assert max(lz.held_out_distance(x) for x in held) < lz.SHAPE_TOL
    assert min(lz.effective_exponent(x) for x in held) > 1.3


def test_zoo_covers_exponents_and_screening():
    zoo, _ = lz.sample_zoo(200, 4738)
    exps = np.array([lz.effective_exponent(x) for x in zoo])
    assert exps.min() < 1.2 and exps.max() > 3.0
    assert any(x[4] > 0 for x in zoo) and any(x[2] != 0 for x in zoo) and any(x[0] < 0 for x in zoo)


def test_params_of_law_matches_scatter_multi_and_batch_generates():
    s = 3.0
    ref = sm.params_of([("pow2", s), None, None, None])
    assert np.allclose(lz.params_of_law(sfar.LAWS["pow2"], s), ref)
    rng = np.random.default_rng(4738)
    d = lz.gen_zoo_batch([(sfar.LAWS["pow2"], 1.0), (lz.sample_law(rng), 1.0)], rng, torch.device("cpu"), 3)
    assert d["P"].shape == (2, 4, sm.N_PAD, 2) and d["params"].shape == (2, 7, 4) and bool(torch.isfinite(d["P"]).all())
    assert d["mask"].sum(1).long().tolist() == d["n"].tolist()


def test_encoder_shape_padding_permutation_and_gradient():
    torch.manual_seed(4738)
    enc = lz.LawEncoder()
    B, N, n = 2, 40, 30
    P = torch.randn(B, 6, N, 2) * 5
    V = torch.randn(B, 6, N, 2)
    mask = torch.zeros(B, N)
    mask[:, :n] = 1.0
    code = enc(P, V, mask)
    assert code.shape == (B, sm.DC) and torch.isfinite(code).all() and (code[:, lz.CODE_DIM:] == 0).all()
    Pb, Vb = P.clone(), V.clone()
    Pb[:, :, n:] = torch.randn(B, 6, N - n, 2) * 50
    Vb[:, :, n:] = torch.randn(B, 6, N - n, 2) * 50
    assert torch.allclose(enc(Pb, Vb, mask), code, atol=1e-5)
    perm = torch.cat([torch.randperm(n), torch.arange(n, N)])
    assert torch.allclose(enc(P[:, :, perm], V[:, :, perm], mask), code, atol=1e-5)
    code.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_encoder_code_depends_on_the_law():
    rng = np.random.default_rng(4738)
    laws = [(sfar.LAWS["pow1"], 1.0), (sfar.LAWS["pow3"], 1.0)]
    d = lz.gen_zoo_batch(laws, rng, torch.device("cpu"), 5)
    torch.manual_seed(4738)
    enc = lz.LawEncoder()
    with torch.no_grad():
        code = enc(d["P"].double(), d["V"].double(), d["mask"].double()).float()
    assert (code[0] - code[1]).abs().max() > 1e-4


def test_exponent_probe_r_squared():
    rng = np.random.default_rng(4738)
    e_tr, e_te = rng.uniform(1, 3.5, 200), rng.uniform(1.2, 2.8, 60)
    f = lambda e: np.stack([e, e ** 2, np.sin(e), np.ones_like(e), e * 0.0], 1)
    assert lz.exponent_probe(f(e_tr) + 0.01 * rng.standard_normal((200, 5)), e_tr, f(e_te) + 0.01 * rng.standard_normal((60, 5)), e_te) > 0.95
    assert lz.exponent_probe(rng.standard_normal((200, 5)), e_tr, rng.standard_normal((60, 5)), e_te) < 0.3


def test_data_check_counts_reconcile_and_resume(tmp_path):
    import json

    from scripts import law_zoo_data_check as dc

    argv = ["--out-dir", str(tmp_path), "--laws", "3", "--scenes-per-law", "1", "--heldout", "2", "--steps", "5", "--device", "cpu"]
    assert dc.main(argv) == 0
    s = json.load(open(tmp_path / "data_summary.json"))
    assert s["requested"] == s["generated"] == 5 and s["accepted"] + s["rejected_by_truth_check"] == 5 and s["encoder_code_finite"]
    mtime = (tmp_path / "chunk000.pt").stat().st_mtime
    assert dc.main(argv) == 0
    assert (tmp_path / "chunk000.pt").stat().st_mtime == mtime
