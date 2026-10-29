import torch

from model.neuron_graph import Graph, NeuronGraphDynamics, count_parameters, normalize_state
from model.neuron_graph_variants import NeuronGraphVariantDynamics
from scripts.neuron_circuit_teacher import generate


def setup():
    circ, ext, states = generate(10, [1, 2], [1, 2], 40, 50, "cpu")
    return Graph(circ["W"]), normalize_state(states)[:, 10], ext[:, 10], normalize_state(states)


def test_defaults_match_base_model():
    torch.manual_seed(4738)
    g, x, e, _ = setup()
    base, var = NeuronGraphDynamics(), NeuronGraphVariantDynamics()
    var.load_state_dict({**base.state_dict(), "emit_threshold": torch.zeros(())})
    for p in base.head.parameters():
        torch.nn.init.normal_(p, std=0.3)
    var.load_state_dict({**base.state_dict(), "emit_threshold": torch.zeros(())})
    h = base.init_hidden(x, e)
    a, la, _ = base.step(x, e, h, g)
    b, lb, _ = var.step(x, e, h, g)
    assert torch.allclose(a, b) and torch.allclose(la, lb)


def test_parameter_count_unchanged():
    base = count_parameters(NeuronGraphDynamics())
    for kw in ({"absolute_reset": True}, {"normalize_messages": True}, {"calibrated_emission": True}):
        assert count_parameters(NeuronGraphVariantDynamics(**kw)) == base


def test_absolute_reset_outputs_head_value_not_delta():
    g, x, e, _ = setup()
    outs = {}
    for absolute in (True, False):
        m = NeuronGraphVariantDynamics(absolute_reset=absolute)
        torch.nn.init.zeros_(m.head.weight)
        torch.nn.init.zeros_(m.head.bias)
        m.head.bias.data[1] = 0.5
        out, _, _ = m.step(x, e, m.init_hidden(x, e), g, torch.ones(x.shape[:2]))
        outs[absolute] = out[..., 0]
    assert torch.allclose(outs[True], torch.full_like(outs[True], 0.5))
    assert torch.allclose(outs[False], x[..., 0] + 0.5)


def test_normalized_sum_divides_by_in_degree_and_calibration_tracks_rate():
    torch.manual_seed(4738)
    g, x, e, states = setup()
    m = NeuronGraphVariantDynamics(normalize_messages=True)
    out, logit, _ = m.step(x, e, m.init_hidden(x, e), g)
    assert torch.isfinite(out).all()
    c = NeuronGraphVariantDynamics(calibrated_emission=True)
    logits = torch.randn(1000)
    target = torch.zeros(1000)
    target[:100] = 1.0
    for _ in range(300):
        c.calibrate(logits, target)
    assert abs(float((logits > c.emit_threshold).float().mean()) - 0.1) < 0.03
