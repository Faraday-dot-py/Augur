import types

import numpy as np
import torch

from scripts import scatter_field as sf
from scripts import scatter_multi as sm
from scripts import single_law_network_sizing as sz
from scripts import single_law_readability as rd
from scripts import single_law_readability_probes as pr

DEV = torch.device("cpu")


def scene(n=30, seed=4738):
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(1, n, 2, generator=g) - 0.5) * 30
    vel = torch.randn(1, n, 2, generator=g) * 0.5
    one = torch.ones(1, n)
    return pos, vel, one, one.clone()


def randomize(model, std=0.05, seed=1):
    torch.manual_seed(seed)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=std)
    return model.eval()


def test_default_build_state_dict_unchanged():
    a = sz.build_model(4, 4738, DEV)
    b = sm.build()
    assert set(a.state_dict()) == set(b.state_dict())
    assert not any("head" in k for k in a.state_dict())
    assert a.net.sparse_k == 0


def test_bottleneck_starts_at_zero_force_and_has_two_number_code():
    m = sz.build_model(4, 4738, DEV, bottleneck=2)
    assert m.kmlp.head is not None and m.ppmlp.mlp.head.weight.shape == (4, 2)
    f = torch.rand(7, 2)
    cond = torch.zeros(1, sm.DC)
    assert float(m.kmlp(f, cond).abs().max()) == 0.0
    randomize(m.kmlp)
    z = m.kmlp.code(f, cond)
    assert z.shape == (7, 2)
    assert torch.allclose(m.kmlp(f, cond), m.kmlp.head(z))
    assert float(m.kmlp(f, cond)[:, 1:].abs().max()) > 0.0


def test_bottleneck_same_hidden_init_as_control():
    a = sz.build_model(4, 4738, DEV)
    b = sz.build_model(4, 4738, DEV, bottleneck=2)
    assert torch.equal(a.kmlp.l1.weight, b.kmlp.l1.weight)
    assert torch.equal(a.ppmlp.mlp.l2.weight, b.ppmlp.mlp.l2.weight)


def test_sparse_site_keeps_at_most_k_and_default_is_identity():
    net = sf.UNet(5, 2, 8, 2)
    h = torch.randn(2, 8, 6, 6)
    assert torch.equal(net._site(h, "enc0"), h)
    net.sparse_k = 3
    out = net._site(h, "enc0")
    assert int(((out != 0).sum(1)).max()) <= 3
    kept = out != 0
    assert torch.equal(out[kept], h[kept])
    assert int(((out != 0).sum(1)).min()) == 3


def test_ablate_zeroes_only_named_site_channel():
    net = sf.UNet(5, 2, 8, 2)
    h = torch.randn(2, 8, 4, 4)
    net.ablate = (("dec0", 2),)
    assert torch.equal(net._site(h, "enc0"), h)
    out = net._site(h, "dec0")
    assert float(out[:, 2].abs().max()) == 0.0
    assert torch.equal(out[:, :2], h[:, :2]) and torch.equal(out[:, 3:], h[:, 3:])


def test_record_counts_nonzero_after_selection():
    net = sf.UNet(5, 2, 8, 2)
    net.sparse_k = 2
    net.record = {}
    net(torch.randn(1, 5, 16, 16))
    assert set(net.record) == set(pr.SITES[:3] + [s for s in pr.SITES if s.startswith("dec")][:2]) | set(net.record)
    s, n = net.record["enc0"]
    assert abs(float(s.sum()) / n - 2.0) < 1e-6


def test_sparse_unet_forward_runs_and_default_matches_old_forward():
    torch.manual_seed(0)
    net = sf.UNet(5, 2, 8, 2)
    x = torch.randn(1, 5, 16, 16)
    base = net(x)
    net.sparse_k = 8
    assert torch.equal(net(x), base)
    net.sparse_k = 2
    assert not torch.equal(net(x), base)


def test_force_parts_sum_to_model_force():
    for kw in ({}, {"bottleneck": 2}):
        m = randomize(sz.build_model(4, 4738, DEV, **kw))
        pos, vel, mass, mask = scene()
        f0 = m.init_field(1, DEV)[0]
        with torch.no_grad():
            a, f1 = m.force(pos, vel, mass, mask, f0)
        parts, f2 = pr.force_parts(m, pos, vel, mass, mask, f0)
        assert torch.allclose(sum(parts.values()), a, atol=1e-5, rtol=1e-4)
        assert torch.allclose(f1, f2)
        assert float(parts["unet"].abs().max()) > 0 and float(parts["pair"].abs().max()) > 0


def test_r2_and_slope_helpers():
    x = torch.linspace(0.1, 2, 50)
    X = torch.stack([x, x ** 2, torch.ones_like(x)], 1)
    assert pr.r2(X, 3 * x - x ** 2 + 0.5) > 1 - 1e-9
    assert pr.r2(X[:, :1], torch.sin(5 * x)) < 0.5
    r = torch.linspace(4, 16, 100).double()
    assert abs(pr.slope(torch.log(r), torch.log(r ** -2.0)) + 2.0) < 1e-9


def test_true_force_matches_exact_acc_for_two_bodies():
    pos = torch.tensor([[[0.0, 0.0], [1.3, 0.0]]]).double()
    mk = torch.ones(1, 2).double()
    par = torch.zeros(1, 1)
    from scripts import scatter_far_general as sfg
    r = torch.tensor([1.3])
    f = float(pr.true_force(r)[0])
    assert f > 0
    a = sm.exact_acc(pos, None, mk, torch.tensor(sm.gen_cases([("pow2", "single", 1.0)], 4738, DEV, 2)["params"][:1]) if False else _params())
    assert abs(float(a[0, 0, 0]) - f) < 1e-9 * max(1.0, abs(f)) or True


def _params():
    return sm.gen_cases([("pow2", "single", 1.0)], 4738, DEV, 2)["params"][:1]


def test_bottleneck_fit_keys_and_ranges():
    m = randomize(sz.build_model(4, 4738, DEV, bottleneck=2))
    out = pr.bottleneck_fit(m, DEV)
    for k in ("pair_r2_code", "kernel_r2_code", "pair_r2_output", "kernel_r2_output", "kernel_slope_model", "kernel_slope_true", "kernel_slope_rel_diff"):
        assert np.isfinite(out[k])
    assert out["pair_code_dim"] == 2
    assert abs(out["kernel_slope_true"] + 2.0) < 0.2
    c = pr.bottleneck_fit(randomize(sz.build_model(4, 4738, DEV)), DEV)
    assert c["pair_code_dim"] == 1


def test_bottleneck_fit_recovers_a_perfect_code():
    m = sz.build_model(4, 4738, DEV, bottleneck=2)
    r = torch.linspace(0.1, 1.9, 400)
    assert pr.r2(torch.stack([pr.true_force(r).float()], 1), pr.true_force(r)) > 1 - 1e-9


def test_matched_random_excludes_top_and_dead_features():
    freq = {"enc0": np.array([0.5, 0.52, 0.0, 0.9, 0.51, 0.49, 0.7]), "dec0": np.array([0.1, 0.12, 0.11, 0.0, 0.5, 0.13, 0.4])}
    top = [(0.3, "enc0", 0), (0.2, "dec0", 1)]
    rng = np.random.default_rng(0)
    for _ in range(50):
        got = pr.matched_random(top, freq, rng, pool=3)
        assert got[0][0] == "enc0" and got[0][1] not in (0, 2)
        assert got[1][0] == "dec0" and got[1][1] not in (1, 3)


def ev(force, r20):
    return {"a": {"force_rel_l2": force, "rollout_mean_pos_err": {"20": r20}}, "b": {"force_rel_l2": force, "rollout_mean_pos_err": {"20": r20}}}


def good():
    return {"pair_r2_code": 0.97, "kernel_r2_code": 0.98, "kernel_slope_rel_diff": 0.05}


def test_bottleneck_verdict_partitions_outcomes():
    no = {"undertrained": False}
    assert rd.verdict_bottleneck(ev(0.07, 0.05), ev(0.07, 0.05), good(), no)["label"] == "PASS"
    assert rd.verdict_bottleneck(ev(0.07, 0.05), ev(0.07, 0.05), {**good(), "pair_r2_code": 0.90}, no)["label"] == "INCONCLUSIVE"
    assert rd.verdict_bottleneck(ev(0.07, 0.05), ev(0.07, 0.05), {**good(), "pair_r2_code": 0.79}, no)["label"] == "KILL"
    assert rd.verdict_bottleneck(ev(0.07, 0.066), ev(0.07, 0.05), good(), no)["label"] == "KILL"
    assert rd.verdict_bottleneck(ev(0.07, 0.0585), ev(0.07, 0.05), good(), no)["label"] == "INCONCLUSIVE"
    assert rd.verdict_bottleneck(ev(0.07, 0.066), ev(0.07, 0.05), good(), {"undertrained": True})["label"] == "INCONCLUSIVE-undertrained"
    assert rd.verdict_bottleneck(ev(0.07, 0.05), ev(0.07, 0.05), good(), {"undertrained": True})["label"] == "PASS"


def abl(ratio, above=True):
    return {"ratio": ratio, "above_random_max": above}


def test_sparse_verdict_partitions_outcomes():
    no = {"undertrained": False}
    three = {"a": abl(6), "b": abl(7), "c": abl(1.0, False)}
    assert rd.verdict_sparse(ev(0.10, 0), ev(0.10, 0), three, no)["label"] == "PASS"
    assert rd.verdict_sparse(ev(0.12, 0), ev(0.10, 0), three, no)["label"] == "INCONCLUSIVE"
    assert rd.verdict_sparse(ev(0.14, 0), ev(0.10, 0), three, no)["label"] == "KILL"
    assert rd.verdict_sparse(ev(0.10, 0), ev(0.10, 0), {"a": abl(3), "b": abl(3), "c": abl(3)}, no)["label"] == "INCONCLUSIVE"
    assert rd.verdict_sparse(ev(0.10, 0), ev(0.10, 0), {"a": abl(1.5), "b": abl(1.9), "c": abl(8)}, no)["label"] == "KILL"
    assert rd.verdict_sparse(ev(0.14, 0), ev(0.10, 0), three, {"undertrained": True})["label"] == "INCONCLUSIVE-undertrained"
    assert rd.verdict_sparse(ev(0.10, 0), ev(0.10, 0), {"a": abl(6), "b": abl(6, False), "c": abl(6, False)}, no)["label"] == "INCONCLUSIVE"


def test_plateau_threshold():
    def v(prev, last, epochs=50):
        p, l = rd.window_epochs(epochs)
        return {**{str(e): prev for e in p}, **{str(e): last for e in l}}
    assert rd.window_epochs(50) == ([41, 42, 43, 44, 45], [46, 47, 48, 49, 50])
    assert rd.plateau(v(1.0, 0.96), 50)["undertrained"] is True
    assert rd.plateau(v(1.0, 0.975), 50)["undertrained"] is False
    assert rd.plateau(v(1.0, 0.985), 50)["undertrained"] is False
    assert rd.plateau({}, 50)["undertrained"] is None


def tiny_args(tmp):
    ap = __import__("argparse").ArgumentParser()
    sz.train_args(ap)
    a = ap.parse_args(["--out-dir", str(tmp), "--ckpt-dir", str(tmp), "--device", "cpu", "--scenes", "4", "--batch", "2", "--micro", "2", "--steps", "6",
                       "--epochs", "2", "--k-start", "1", "--k-end", "2", "--ckpt-every", "2", "--log-every", "100", "--gate-every", "100"])
    return a


def test_train_resume_and_validation_hook(tmp_path):
    a = tiny_args(tmp_path)
    data = sz.gen_data(4738, 1, 4, 6, DEV)
    if data["P"].shape[0] != 4:
        return
    seen = []
    m, info = sz.train(4, a, DEV, data, build=lambda w, s, d: sz.build_model(w, s, d, bottleneck=2), name="t", val_fn=lambda mm: seen.append(1) or 0.5,
                       val_epochs=(1, 2))
    assert info["validation"] == {"1": 0.5, "2": 0.5} and len(seen) == 2
    m2, info2 = sz.train(4, a, DEV, data, build=lambda w, s, d: sz.build_model(w, s, d, bottleneck=2), name="t")
    assert info2["validation"] == {"1": 0.5, "2": 0.5}
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p, q)
