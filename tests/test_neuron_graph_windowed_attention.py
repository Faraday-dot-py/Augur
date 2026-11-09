import torch

from model.neuron_graph import Graph, normalize_state
from model.neuron_graph_windowed_attention import WindowedAttentionDynamics
from scripts.neuron_circuit_teacher import generate
from scripts.train_neuron_graph_smoke import sequence_loss


def setup():
    circ, ext, states = generate(10, [1, 2], [1, 2], 80, 50, "cpu")
    return Graph(circ["W"]), normalize_state(states), ext


def randomize(m):
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3) if p.dim() > 1 else None


def test_causal_window_ignores_older_than_window():
    torch.manual_seed(4738)
    g, st, ext = setup()
    m = WindowedAttentionDynamics(window=4)
    randomize(m)
    outs = []
    for first in (st[:, 0], st[:, 20]):
        h = m.init_hidden(first, ext[:, 0])
        x = first
        for k, src in enumerate([first] + [st[:, 30 + i] for i in range(7)]):
            out, _, h = m.step(src, ext[:, 30 + k], h, g)
        outs.append(out)
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def test_first_tick_attends_only_to_itself_and_loss_backward():
    torch.manual_seed(4738)
    g, st, ext = setup()
    m = WindowedAttentionDynamics()
    randomize(m)
    h = m.init_hidden(st[:, 0], ext[:, 0])
    out, _, h = m.step(st[:, 0], ext[:, 0], h, g)
    assert torch.isfinite(out).all() and h[0, -1] == 1
    loss = sequence_loss(m, g, st[:, :17], ext[:, :16], False)
    loss.backward()
    assert m.attn[1].q.weight.grad is not None and torch.isfinite(loss)
