import torch

from model.neuron_graph import Graph, LinearDynamics, NeuronGraphDynamics, PerCellMlpDynamics, count_parameters, linear_features, normalize_state
from scripts.neuron_circuit_teacher import generate, make_circuit
from scripts.train_neuron_graph_smoke import free_rollout, rollout_metrics, raw_state, sequence_loss


def tiny(n=10, c=3, steps=40):
    circ, ext, states = generate(n, list(range(c)), list(range(c)), steps, 50, "cpu")
    return circ, ext, normalize_state(states)


def test_teacher_finite_signed_and_sparse():
    circ, ext, states = generate(30, [1, 2], [1, 2], 80, 50, "cpu")
    assert torch.isfinite(states).all()
    assert states[..., 0].max() <= 30.0 and states[..., 0].min() > -150.0
    w = circ["W"]
    assert (w[0].diagonal() == 0).all()
    assert (w > 0).any() and (w < 0).any()
    assert ((w[0] != 0).float().mean() < 0.5)
    exc = circ["exc"][0]
    assert (w[0][:, exc] < 0).sum() == 0 and (w[0][:, ~exc] > 0).sum() == 0
    assert states[..., 3].sum() > 0


def test_circuits_differ_by_seed_and_repeat():
    a, b, c = make_circuit(20, 5), make_circuit(20, 6), make_circuit(20, 5)
    assert not torch.equal(a["W"], b["W"])
    assert torch.equal(a["W"], c["W"])


def test_graph_edges_match_dense():
    circ, _, _ = tiny()
    g = Graph(circ["W"])
    dense = torch.zeros_like(circ["W"]).reshape(-1)
    n = g.n
    b = g.dst // n
    dense = torch.zeros_like(circ["W"])
    dense[b, g.dst % n, g.src % n] = g.w
    assert torch.equal(dense, circ["W"])


def test_messages_depend_on_presynaptic_cells_only_when_enabled():
    torch.manual_seed(4738)
    circ, ext, st = tiny()
    g = Graph(circ["W"])
    src, dst = int(g.src[0]), int(g.dst[0])
    for messages, expect_change in ((True, True), (False, False)):
        model = NeuronGraphDynamics(messages=messages)
        for p in model.head.parameters():
            torch.nn.init.normal_(p, std=0.3)
        x = st[:, 5]
        h = model.init_hidden(x, ext[:, 5])
        base, _, _ = model.step(x, ext[:, 5], h, g)
        x2 = x.clone().reshape(-1, 4)
        x2[src] = x2[src] + 1.0
        h2 = h.clone()
        h2[src] = h2[src] + 1.0
        pert, _, _ = model.step(x2.reshape(x.shape), ext[:, 5], h2, g)
        changed = (base.reshape(-1, 4)[dst] - pert.reshape(-1, 4)[dst]).abs().max() > 1e-6
        assert bool(changed) == expect_change


def test_equal_parameters_with_and_without_messages():
    assert count_parameters(NeuronGraphDynamics(messages=True)) == count_parameters(NeuronGraphDynamics(messages=False))


def test_cell_permutation_equivariance():
    torch.manual_seed(4738)
    circ, ext, st = tiny(n=8, c=1)
    model = NeuronGraphDynamics()
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=0.3)
    perm = torch.randperm(8)
    w = circ["W"]
    wp = w[:, perm][:, :, perm]
    x, e = st[:, 7], ext[:, 7]
    h = model.init_hidden(x, e)
    out, _, _ = model.step(x, e, h, Graph(w))
    outp, _, _ = model.step(x[:, perm], e[:, perm], h[perm], Graph(wp))
    assert torch.allclose(out[:, perm], outp, atol=1e-5)


def test_losses_finite_and_backprop_both_stages():
    torch.manual_seed(4738)
    circ, ext, st = tiny()
    g = Graph(circ["W"])
    for model in (NeuronGraphDynamics(), PerCellMlpDynamics()):
        for rollout in (False, True):
            loss = sequence_loss(model, g, st[:, :9], ext[:, :8], rollout)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_linear_fit_and_metrics_shapes():
    circ, ext, st = tiny(c=2, steps=60)
    g = Graph(circ["W"])
    lin = LinearDynamics()
    feats = torch.stack([linear_features(st[:, t], ext[:, t], g) for t in range(50)])
    lin.fit(feats, st[:, 1:51].transpose(0, 1))
    x0 = st[:, 0]
    pred = raw_state(free_rollout(lin, x0, ext, g, 40))
    truth = raw_state(st[:, 1:41])
    res = rollout_metrics(pred, truth, 1, 2)
    assert res["err@10"].shape == (1,) and 0.0 <= res["spike_f1@20"][0] <= 1.0
    perfect = rollout_metrics(truth, truth, 1, 2)
    assert perfect["err@10"][0] == 0.0 and perfect["spike_f1@20"][0] == 1.0
