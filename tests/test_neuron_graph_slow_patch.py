import torch

from model.neuron_graph import Graph, normalize_state
from model.neuron_graph_slow_patch import PATCH, SlowPatchTokenDynamics
from scripts.neuron_circuit_teacher import generate
from scripts.train_neuron_graph_slow_patch import sequence_loss


def setup():
    circ, ext, states = generate(10, [1, 2], [1, 2], 80, 50, "cpu")
    return Graph(circ["W"]), normalize_state(states), ext


def test_count_pred_only_at_patch_boundary_and_state_resets():
    torch.manual_seed(4738)
    g, st, ext = setup()
    m = SlowPatchTokenDynamics()
    x = st[:, 0]
    h = m.init_hidden(x, ext[:, 0])
    seen = []
    for k in range(2 * PATCH + 3):
        x, _, h = m.step(x, ext[:, k], h, g)
        seen.append(m.count_pred is not None)
    assert [i for i, s in enumerate(seen) if s] == [PATCH - 1, 2 * PATCH - 1]
    assert h[0, -1] == 3 and torch.isfinite(h).all()


def test_slow_token_changes_after_patch_and_feeds_back():
    torch.manual_seed(4738)
    g, st, ext = setup()
    m = SlowPatchTokenDynamics()
    for p in m.slow_update.parameters():
        torch.nn.init.normal_(p, std=0.5)
    x = st[:, 0]
    h = m.init_hidden(x, ext[:, 0])
    for k in range(PATCH):
        x, _, h = m.step(x, ext[:, k], h, g, st[:, k + 1, :, 3])
    slow = h[:, m.hidden:m.hidden + m.slow_dim]
    assert slow.abs().sum() > 0


def test_loss_backward_reaches_count_head():
    torch.manual_seed(4738)
    g, st, ext = setup()
    m = SlowPatchTokenDynamics()
    loss = sequence_loss(m, g, st[:, :49], ext[:, :48], False)
    loss.backward()
    assert m.slow_update[0].weight.grad is not None and torch.isfinite(loss)
