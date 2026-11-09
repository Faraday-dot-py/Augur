import torch

from model.neuron_graph import Graph
from model.neuron_graph import count_parameters
from model.neuron_graph_plastic import PlasticNeuronGraphDynamics, normalize_plastic_state
from scripts.neuron_plasticity_teacher import (
    CONDITIONS, PROBE_TICKS, build_ext, make_plastic_circuit, pick_synapse, probe_response, run_episodes,
)
from scripts.train_neuron_plasticity_smoke import sequence_loss, window


def episode(delta, n=10, seed=3):
    c = make_plastic_circuit(n, seed)
    pre, post = pick_synapse(c, torch.Generator().manual_seed(seed))
    ext = build_ext(n, pre, post, delta, seed)
    circ, (states, gh) = run_episodes([c], [ext], "cpu")
    return states, gh, pre, post, circ


def test_stdp_sign_follows_spike_order():
    final = {}
    for cond, delta in CONDITIONS.items():
        states, gh, pre, post, _ = episode(delta)
        final[cond] = float(gh[0, -1, post, pre])
    assert final["forward"] > 1.3 and final["reversed"] < 0.7 and abs(final["none"] - 1.0) < 0.05


def test_probe_response_moves_with_plasticity():
    fwd, _, _, post, _ = episode(5)
    rev, _, _, post2, _ = episode(-5)
    r = lambda s, p: [float(probe_response(s[0, :, p, 0], s[0, :, p, 3], p, t)) for t in PROBE_TICKS]
    assert r(fwd, post)[1] > r(fwd, post)[0] and r(rev, post2)[1] < r(rev, post2)[0]


def test_inhibitory_synapses_do_not_change():
    states, gh, pre, post, circ = episode(5)
    inh = (circ["W"][0] != 0) & ~circ["exc"][0][None, :]
    assert torch.all(gh[0, -1][inh] == 1.0)


def test_fixed_arm_never_updates_synapse_state_and_equal_params():
    torch.manual_seed(4738)
    states, gh, pre, post, circ = episode(5)
    g = Graph(circ["W"])
    x = normalize_plastic_state(states[:, 50])
    ext = torch.zeros(1, 10)
    m0 = torch.zeros(g.w.shape[0])
    plastic, fixed = PlasticNeuronGraphDynamics(plastic=True), PlasticNeuronGraphDynamics(plastic=False)
    assert count_parameters(plastic) == count_parameters(fixed)
    for model in (plastic, fixed):
        for p in model.rule.parameters():
            torch.nn.init.normal_(p, std=0.5)
    h = fixed.init_hidden(x, ext)
    _, _, _, m1 = fixed.step(x, ext, h, m0, g)
    assert torch.equal(m1, m0)
    _, _, _, m2 = plastic.step(x, ext, plastic.init_hidden(x, ext), m0, g)
    assert not torch.equal(m2, m0)


def test_losses_backprop_for_both_arms():
    torch.manual_seed(4738)
    c = [make_plastic_circuit(10, s) for s in range(2)]
    exts = [build_ext(10, *pick_synapse(x, torch.Generator().manual_seed(1)), 5, 1) for x in c]
    circ, (states, gh) = run_episodes(c, exts, "cpu")
    data = {10: (circ["W"], torch.stack(exts), normalize_plastic_state(states), gh)}
    gen = torch.Generator().manual_seed(0)
    graph, st, ex, m_true = window(data, 10, 2, 8, gen, "cpu")
    assert m_true.shape == (9, graph.w.shape[0])
    for plastic in (True, False):
        model = PlasticNeuronGraphDynamics(plastic=plastic)
        for rollout in (False, True):
            loss = sequence_loss(model, graph, st, ex, m_true, rollout)
            loss.backward()
            assert torch.isfinite(loss)
